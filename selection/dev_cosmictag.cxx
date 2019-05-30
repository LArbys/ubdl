#include <iostream>
#include <map>
#include <utility>
// ROOT
#include "TFile.h"
#include "TTree.h"
#include "TH2D.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TGraph.h"

// larlite
#include "DataFormat/storage_manager.h"
#include "DataFormat/larflow3dhit.h"
#include "DataFormat/larflowcluster.h"
#include "DataFormat/opflash.h"
#include "DataFormat/mctrack.h"
#include "DataFormat/mcshower.h"
#include "DataFormat/mctruth.h"
#include "DataFormat/mcnu.h"
// larutil
#include "LArUtil/LArProperties.h"
#include "LArUtil/DetectorProperties.h"
#include "LArUtil/Geometry.h"
#include "LArUtil/ClockConstants.h"
#include "LArUtil/SpaceChargeMicroBooNE.h"
// larcv
#include "larcv/core/DataFormat/IOManager.h"
#include "larcv/core/DataFormat/EventImage2D.h"
#include "larcv/core/DataFormat/EventChStatus.h"
#include "larcv/core/DataFormat/EventClusterMask.h"
#include "larcv/core/DataFormat/EventPGraph.h"
#include "larcv/core/DataFormat/EventROI.h"

/**
 *
 * cosmic tag/rejection
 *
 */
int is_clust_out_bounds(int early_count, int late_count, int threshold=0);
int is_cluster_bkgdfrac_cut(float bkgdfrac, float threshold);
int is_cluster_showerfrac_cut(float showerfrac, float threshold);
int is_cluster_outsidefrac_cut(float outsidefrac, float threshold);
float calc_dWall(float x, float y, float z);
std::vector<int> getProjectedPixel(const std::vector<float>& pos3d,
				   const larcv::ImageMeta& meta,
				   const int nplanes,
				   const float fracpixborder=1.5 );

struct pos{
  float x=-1;
  float y=-1;
  float z=-1;
  float dWall=-1;
  int idx=-1;

};


int main( int nargs, char** argv ) {

  gStyle->SetOptStat(1);
  gStyle->SetStatY(0.4);
  gStyle->SetStatX(0.30);

  std::string supera  = argv[1];
  std::string mcinfo = argv[2];
  std::string ubmrcnn = argv[3];
  std::string ubclust = argv[2];
  std::string larcvtruth = argv[3];
  std::string ssnet = argv[4];
  std::string vtx = argv[5];
  std::string vtx2 = argv[6];
  //std::string infill = argv[7];
  //std::string outfile_larlite = argv[5];
  std::string outfile_larcv = argv[7];

  std::string saveme = "";
  std::string calc_eff="";
  
  if(nargs>8){
    saveme = argv[8];
    calc_eff = argv[8];
    if(nargs>9) saveme = argv[9];
  }
  
  bool saveana=false;
  if(saveme=="save") saveana=true;
  bool doeff=false;
  if(calc_eff=="eff") doeff=true; 
  
  // ADC
  larcv::IOManager io_supera( larcv::IOManager::kREAD, "", larcv::IOManager::kTickBackward );
  io_supera.add_in_file( supera );
  io_supera.initialize();

  // MCTruth
  larlite::storage_manager io_mcinfo( larlite::storage_manager::kREAD );
  io_mcinfo.add_in_filename( mcinfo );
  io_mcinfo.open();

  // Larflowclusters
  larlite::storage_manager io_ubclust( larlite::storage_manager::kREAD );
  io_ubclust.add_in_filename( ubclust );
  io_ubclust.open();

  // Cluster masks
  larcv::IOManager io_ubmrcnn( larcv::IOManager::kREAD, "", larcv::IOManager::kTickBackward );
  io_ubmrcnn.add_in_file( ubmrcnn );
  io_ubmrcnn.initialize();

  // Instance & True ROI
  larcv::IOManager io_larcvtruth( larcv::IOManager::kREAD, "", larcv::IOManager::kTickBackward );
  io_larcvtruth.add_in_file( larcvtruth );
  io_larcvtruth.initialize();

  // SSNet
  larcv::IOManager io_ssnet( larcv::IOManager::kREAD, "", larcv::IOManager::kTickBackward );
  io_ssnet.add_in_file( ssnet );
  io_ssnet.initialize();

  // Reco vtx from PGraph (w/out tagger)
  larcv::IOManager io_vtx( larcv::IOManager::kREAD, "", larcv::IOManager::kTickBackward );
  io_vtx.add_in_file( vtx );
  io_vtx.initialize();

  // Reco vtx from PGraph (w/ tagger)
  larcv::IOManager io_vtx2( larcv::IOManager::kREAD, "", larcv::IOManager::kTickBackward );
  io_vtx2.add_in_file( vtx2 );
  io_vtx2.initialize();

  // output
  larcv::IOManager out_larcv( larcv::IOManager::kWRITE, "", larcv::IOManager::kTickForward );
  out_larcv.add_in_file( outfile_larcv );
  out_larcv.initialize();
  /*
  larlite::storage_manager out_larlite( larlite::storage_manager::kWRITE );
  out_larlite.set_out_filename( outfile_larlite );
  out_larlite.open();
  */

  // Constants
  const float drift_vel = larutil::LArProperties::GetME()->DriftVelocity();
  const float width = larutil::Geometry::GetME()->DetHalfWidth() * 2;
  const float time_per_tick = 1/larutil::kDEFAULT_FREQUENCY_TPC;
  std::cout << "Width is         " << width << std::endl;
  std::cout << "Drift Vel is     " << drift_vel << std::endl;
  std::cout << "Time per tick " << time_per_tick << std::endl;

  float beam_low = 3200.;
  float beam_high = 3200 + (width / drift_vel ) / time_per_tick; //width/drift speed / us per tick
  std::cout << "Beam Low  is  " << beam_low << std::endl;
  std::cout << "Beam High is "  << beam_high << std::endl;

  // thresholds for ssnet pixel classification
  const float ssnet_bkgd_score_tresh = 0.5;
  const float ssnet_shower_score_tresh = 0.5;
  const float ssnet_track_score_tresh = 0.5;

  const float bkgd_frac_cut = 0.3;
  const float shower_frac_cut = 0.1;
  const float outside_frac_cut = 0.1;

  // dist to detector boundary threshold in cm
  const float dwall_thresh = 10.0;

  //nu fraction for cluster neutrino labeling
  const float nufrac_thresh = 0.5;

  // Ana tree
  TFile* fout = new TFile("cosmictag_new.root","recreate");
  TTree* tree = NULL;

  int run=-1;
  int subrun=-1;
  int event=-1;

  int interaction_type;
  int interaction_mode;
  int ccnc;
  int num_nue;
  int num_numu;
  int num_good_vtx;
  int num_bad_vtx;
  int num_good_vtx_tagger;
  int num_bad_vtx_tagger;

  std::vector<float> PrimaryPEnergy; // deposited energy of primary p
  std::vector<float> PrimaryEEnergy; // deposited energy of primary e
  std::vector<float> PrimaryMuEnergy; // deposited energy of primary mu

  std::vector<float> _scex(3,0);
  std::vector<float> _tx(3,0);
  std::vector<int> nupixel; //tick u v y wire
  std::vector<int> scenupixel;

  std::vector<std::vector<float> > good_vtx;
  std::vector<std::vector<float> > bad_vtx;
  std::vector<std::vector<float> > good_vtx_tag;
  std::vector<std::vector<float> > bad_vtx_tag;
  std::vector<std::vector<float> > good_vtx_pixel;
  std::vector<std::vector<float> > bad_vtx_pixel;
  std::vector<std::vector<float> > good_vtx_tag_pixel;
  std::vector<std::vector<float> > bad_vtx_tag_pixel;

  // final state particles
  int NumPiPlus;
  int NumPiMinus;
  int NumPi0;
  int NumProton;
  int NumNeutron;
  int NumGamma;
  int NumElectron;
  int NumMuon;

  std::vector<int> FinalStatePDG; // Note: in NC the neutrino is the outgoing lepton

  // ADC images
  // before DL tagger
  TH2F* hwoTagger[3];
  hwoTagger[0] = new TH2F("hUwoTagger","w/o CR tagger;U wire; time tick",3456,0.,3456,1008,0.,1008.);
  hwoTagger[1] = new TH2F("hVwoTagger","w/o CR tagger;V wire; time tick",3456,0.,3456,1008,0.,1008.);
  hwoTagger[2] = new TH2F("hYwoTagger","w/o CR tagger;Y wire; time tick",3456,0.,3456,1008,0.,1008.);

  // a plethora of TH2Fs
  TH2F* hclusterid = new TH2F("hclusterid","Cluster ID;Y wire; time tick",3456,0.,3456,1008,0.,1008.);
  TH2F* hnufrac = new TH2F("hnufrac","Nu frac;Y wire; time tick",3456,0.,3456,1008,0.,1008.);
  TH2F* hcroi = new TH2F("hcroi","Outside cROI frac;Y wire; time tick",3456,0.,3456,1008,0.,1008.);
  TH2F* hssnetbg = new TH2F("hssnetbg","SSNet BG frac;Y wire; time tick",3456,0.,3456,1008,0.,1008.);
  TH2F* hssnetshower = new TH2F("hssnetshower","SSNet shower frac;Y wire; time tick",3456,0.,3456,1008,0.,1008.);
  TH2F* hssnettrack = new TH2F("hssnettrack","SSNet track frac;Y wire; time tick",3456,0.,3456,1008,0.,1008.);
  TH2F* hearly = new TH2F("hearly","early hits;Y wire; time tick",3456,0.,3456,1008,0.,1008.);
  TH2F* hlate = new TH2F("hlate","late hits;Y wire; time tick",3456,0.,3456,1008,0.,1008.);
  TH2F* hdwall = new TH2F("hdwall","outside fiducial hits;Y wire; time tick",3456,0.,3456,1008,0.,1008.);
  
  TH2F* eff_hist = NULL;
  if(doeff){
    eff_hist = new TH2F("eff_hist", "Nu Efficiency, Cosmic Rejection, Pixelwise", 20,0.0,1.001,20,0.0,1.001);
    eff_hist->SetOption("COLZ");
    eff_hist->SetXTitle("Neutrino Efficiency");
    eff_hist->SetYTitle("Cosmic Rejection");
  }
  //

  if(saveana){
    tree = new TTree("tree","event tree");
    tree->Branch("hUwoTagger","TH2F",&hwoTagger[0]);
    tree->Branch("hVwoTagger","TH2F",&hwoTagger[1]);
    tree->Branch("hYwoTagger","TH2F",&hwoTagger[2]);
    tree->Branch("hclusterid","TH2F",&hclusterid);
    tree->Branch("hnufrac","TH2F",&hnufrac);
    tree->Branch("hcroi","TH2F",&hcroi);
    tree->Branch("hssnetbg","TH2F",&hssnetbg);
    tree->Branch("hssnetshower","TH2F",&hssnetshower);
    tree->Branch("hssnettrack","TH2F",&hssnettrack);
    tree->Branch("hearly","TH2F",&hearly);
    tree->Branch("hlate","TH2F",&hlate);
    tree->Branch("hdwall","TH2F",&hdwall);	
    tree->Branch("run",&run,"run/I");
    tree->Branch("subrun",&subrun,"subrun/I");
    tree->Branch("event",&event,"event/I");
    tree->Branch("interaction_type",&interaction_type,"interaction_type/I");
    tree->Branch("interaction_mode",&interaction_mode,"interaction_mode/I");
    tree->Branch("CCNC",&ccnc,"CCNC/I");
    tree->Branch("FinalStatePDG",&FinalStatePDG);
    tree->Branch("num_piplus",&NumPiPlus,"num_piplus/I");
    tree->Branch("num_piminus",&NumPiMinus,"num_piminus/I");
    tree->Branch("num_pi0",&NumPi0,"num_pi0/I");
    tree->Branch("num_p",&NumProton,"num_p/I");
    tree->Branch("num_n",&NumNeutron,"num_n/I");
    tree->Branch("num_e",&NumElectron,"num_e/I");
    tree->Branch("num_mu",&NumMuon,"num_mu/I");
    tree->Branch("num_gamma",&NumGamma,"num_gamma/I");
    tree->Branch("num_nue",&num_nue,"num_nue/I");
    tree->Branch("num_numu",&num_numu,"num_numu/I");
    tree->Branch("p_enedep",&PrimaryPEnergy);
    tree->Branch("e_enedep",&PrimaryEEnergy);
    tree->Branch("mu_enedep",&PrimaryMuEnergy);
    tree->Branch("num_good_vtx",&num_good_vtx,"num_good_vtx/I");
    tree->Branch("num_bad_vtx",&num_bad_vtx,"num_bad_vtx/I");
    tree->Branch("num_good_vtx_tagger",&num_good_vtx_tagger,"num_good_vtx_tagger/I");
    tree->Branch("num_bad_vtx_tagger",&num_bad_vtx_tagger,"num_bad_vtx_tagger/I");
    tree->Branch("_tx",&_tx);
    tree->Branch("_scex",&_scex);
    tree->Branch("nupixel",&nupixel);
    tree->Branch("scenupixel",&scenupixel);
    tree->Branch("good_vtx", &good_vtx);
    tree->Branch("good_vtx_pixel", &good_vtx_pixel);
    tree->Branch("bad_vtx", &good_vtx);
    tree->Branch("bad_vtx_pixel", &bad_vtx_pixel);
    tree->Branch("good_vtx_tag", &bad_vtx_tag);
    tree->Branch("good_vtx_tag_pixel", &good_vtx_tag_pixel);
    tree->Branch("bad_vtx_tag", &bad_vtx_tag);
    tree->Branch("bad_vtx_tag_pixel", &bad_vtx_tag_pixel);
  }

  ::larutil::SpaceChargeMicroBooNE* sce = new ::larutil::SpaceChargeMicroBooNE;

  /********* EVENT LOOP *************/
  int nentries = io_supera.get_n_entries();
  //nentries = 10; //temp
  for (int i=0; i<nentries; i++) {

    io_supera.read_entry(i);
    io_mcinfo.go_to(i);
    io_ubclust.go_to(i);
    io_ubmrcnn.read_entry(i);
    io_larcvtruth.read_entry(i);
    io_ssnet.read_entry(i);
    io_vtx.read_entry(i);
    io_vtx2.read_entry(i);
    
    std::cout << "entry " << i << std::endl;
    //************ INITIALIZATION *********//
    std::vector<float> _x(3,0);
    std::vector<int> pixel;
    std::vector<float> floatpixel; //cast int->float
    _tx.clear();
    _scex.clear();
    nupixel.clear();
    scenupixel.clear();
    good_vtx.clear();
    bad_vtx.clear();
    good_vtx_tag.clear();
    bad_vtx_tag.clear();
    good_vtx_pixel.clear();
    bad_vtx_pixel.clear();
    good_vtx_tag_pixel.clear();
    bad_vtx_tag_pixel.clear();
    num_nue=0;
    num_numu=0;
    NumPiPlus=0;
    NumPiMinus=0;
    NumPi0=0;
    NumProton=0;
    NumNeutron=0;
    NumElectron=0;
    NumMuon=0;
    NumGamma=0;
    FinalStatePDG.clear();
    PrimaryPEnergy.clear();
    PrimaryEEnergy.clear();
    PrimaryMuEnergy.clear();
    for(int ii=0; ii<3; ii++){
      hwoTagger[ii]->Reset();
    }
    hclusterid->Reset();
    hnufrac->Reset();
    hcroi->Reset();
    hssnetbg->Reset();
    hssnetshower->Reset();
    hssnettrack->Reset();
    hearly->Reset();
    hlate->Reset();
    hdwall->Reset();
    
    // in
    const auto ev_img            = (larcv::EventImage2D*)io_supera.get_data( larcv::kProductImage2D, "wire" );
    //auto ev_chstatus       = (larcv::EventChStatus*)io_supera.get_data( larcv::kProductChStatus,"wire");
    const auto ev_cluster        = (larlite::event_larflowcluster*)io_ubclust.get_data(larlite::data::kLArFlowCluster,  "rawmrcnn" );
    const auto& ev_mctrack       = *((larlite::event_mctrack*)io_mcinfo.get_data(larlite::data::kMCTrack,  "mcreco" ));
    const auto& ev_mcshower      = *((larlite::event_mcshower*)io_mcinfo.get_data(larlite::data::kMCShower,  "mcreco" ));    
    const auto ev_partroi        = (larcv::EventROI*)(io_larcvtruth.get_data( larcv::kProductROI,"segment"));
    const auto ev_pgraph         = (larcv::EventPGraph*) io_vtx.get_data( larcv::kProductPGraph,"test");
    const auto ev_pgraph2        = (larcv::EventPGraph*) io_vtx2.get_data( larcv::kProductPGraph,"test");
    const auto ev_mctruth        = (larlite::event_mctruth*)io_mcinfo.get_data(larlite::data::kMCTruth,  "generator" );
    //auto ev_opflash_beam   = (larlite::event_opflash*)io_ubpost_larlite.get_data(larlite::data::kOpFlash, "simpleFlashBeam" );
    //auto ev_opflash_cosmic = (larlite::event_opflash*)io_ubpost_larlite.get_data(larlite::data::kOpFlash, "simpleFlashCosmic" );
    //auto ev_ancestor       = (larcv::EventImage2D*)io_larcvtruth.get_data( larcv::kProductImage2D, "ancestor" );
    const auto ev_instance       = (larcv::EventImage2D*)io_larcvtruth.get_data( larcv::kProductImage2D, "instance" );
    const auto ev_ssnet          = (larcv::EventImage2D*)io_ssnet.get_data( larcv::kProductImage2D, "uburn_plane2" );
    const auto ev_clustermask    = (larcv::EventClusterMask*)io_ubmrcnn.get_data( larcv::kProductClusterMask, "mrcnn_masks");

    //out
    //auto evout_cluster     = (larlite::event_larflowcluster*)out_larlite.get_data(larlite::data::kLArFlowCluster, "cosmictag");
    
    run    = io_supera.event_id().run();
    subrun = io_supera.event_id().subrun();
    event  = io_supera.event_id().event();
    std::cout <<"run "<< run <<" subrun " << subrun <<" event " << event << std::endl;

    // some flags for clusters
    std::vector<int> is_thrumu(ev_cluster->size(),-1); //is cluster crossing two TPC faces?
    std::vector<int> is_crossmu(ev_cluster->size(),-1); //is cluster crossing one TPC face?
    std::vector<int> out_of_time(ev_cluster->size(),-1); //is cluster partially outside of beam window?
    std::vector<float> nufrac(ev_cluster->size(),0.); //how many cluster pixels are neutrino?
    std::vector<float> numhits(ev_cluster->size(),0.); //tot. number of hits in cluster
    std::vector<float> showerfrac(ev_cluster->size(),0.); //what fraction are shower pixels?
    std::vector<float> bkgdfrac(ev_cluster->size(),0.);//what fraction are bkgd pixels?
    std::vector<float> outsidefrac(ev_cluster->size(),0.);//what fraction are outside croi?
    std::vector<float> trackfrac(ev_cluster->size(),0.);//what fraction are track pixels?
    std::vector<int> outside_flag(ev_cluster->size(),0.);//what fraction are track pixels?
    std::vector<int> shower_flag(ev_cluster->size(),0.);//what fraction are track pixels?
    std::vector<int> bkgd_flag(ev_cluster->size(),0.);//what fraction are track pixels?

    // save index: if not vetoed we save cluster at idx
    std::vector<int> save_idx;

    //grab Y plane wire image
    auto const& wire_img = ev_img->Image2DArray();
    auto const& wire_meta = wire_img.at(2).meta();

    //grab Y plane ancestor image: obsolete
    //auto const& anc_img = ev_ancestor->Image2DArray().at( 2 );
    //auto const& anc_meta = anc_img.meta();

    //grab Y plane instance image
    auto const& inst_img = ev_instance->Image2DArray().at( 2 );
    auto const& inst_meta = inst_img.meta();

    //grab Y plane SSNet images
    auto const& track_img = ev_ssnet->Image2DArray().at( 0 );
    auto const& shower_img = ev_ssnet->Image2DArray().at( 1 );
    auto const& track_meta = track_img.meta();

    // binarized ADC & instance
    larcv::Image2D binarized_adc = wire_img.at( 2 );
    binarized_adc.binary_threshold(10.0, 0.0, 1.0);
    larcv::Image2D binarized_instance =  inst_img;
    binarized_instance.binary_threshold(-1.0, 0.0, 1.0);
    binarized_instance.eltwise(binarized_adc);


    /*********** PIXEL EFFICIENCY BLOCK ********/
    TH2F* image_hist0 = NULL;
    TH2F* image_hist1 = NULL;
    if(doeff){
      std::string str_title = "Y-Plane All Run:" +std::to_string(run) + " Event: " + std::to_string(event);
      image_hist0 = new TH2F(Form("Y_Plane_all_run%d_event%d",run,event), str_title.c_str(),
			     wire_meta.rows(), 0.0, wire_meta.rows(), wire_meta.cols(), 0.0, wire_meta.cols());
      
      image_hist1 = new TH2F(Form("Y_Plane_cuts_run%d_event%d",run,event), str_title.c_str(),
			     wire_meta.rows(), 0.0, wire_meta.rows(), wire_meta.cols(), 0.0, wire_meta.cols());
      image_hist0->SetOption("COLZ");
      image_hist1->SetOption("COLZ");
      image_hist1->SetTitle("Y-Plane Cuts");
    }
    
    float orig_sum_nu_pix = 0;
    float final_sum_nu_pix = 0;
    float orig_sum_cosm_pix = 0;
    float final_sum_cosm_pix = 0;

    //only run if we want to calculate efficiency
    if(doeff){
      for (int r=0;r<wire_meta.rows();r++){
	for (int c=0;c<wire_meta.cols();c++){
	  if (binarized_instance.pixel(r,c) == 1){
	    image_hist0->SetBinContent(r+1,c+1,2);
	    image_hist1->SetBinContent(r+1,c+1,2);
	    orig_sum_nu_pix++;
	  }
	  else if(binarized_adc.pixel(r,c) ==1){
	    image_hist0->SetBinContent(r+1,c+1,1);
	    image_hist1->SetBinContent(r+1,c+1,1);
	    orig_sum_cosm_pix++;
	  }	  
	}
      }
      // final pixels set to start pixels; will subract later
      final_sum_nu_pix = orig_sum_nu_pix;
      final_sum_cosm_pix = orig_sum_cosm_pix;
    }

    /****** MCTRUTH *********/
    // only one neutrino interaction per entry
    auto& neutrinoinfo = ev_mctruth->at(0).GetNeutrino();
    interaction_mode = neutrinoinfo.Mode();
    interaction_type = neutrinoinfo.InteractionType();
    ccnc = neutrinoinfo.CCNC();
    if(neutrinoinfo.Nu().PdgCode()==12) num_nue++;
    else if(neutrinoinfo.Nu().PdgCode()==14) num_numu++;

    const std::vector<larlite::mcpart> partlist = ev_mctruth->at(0).GetParticles();
    for(int ipart=1; ipart< partlist.size(); ipart++){ // i=0 is incoming neutrino
      int pdg = partlist[ipart].PdgCode();
      int status = partlist[ipart].StatusCode(); // status==1 is final state
      if( status!=1 || pdg==2000000101) continue; // skip non-final state & bindino
      FinalStatePDG.push_back( pdg );
      if(pdg==11 || pdg==-11 ) NumElectron++;
      else if(pdg==13 || pdg==-13 ) NumMuon++;
      else if(pdg==22) NumGamma++;
      else if(pdg==111) NumPi0++;
      else if(pdg==211) NumPiPlus++;
      else if(pdg==-211) NumPiMinus++;
      else if(pdg==2212) NumProton++;
      else if(pdg==2112) NumNeutron++;
    }

    // fill deposited energy of primary protons & muon
    for(auto const& track: ev_mctrack){
      if(track.Origin() != 1) continue; // not neutrino-induced
      if(track.TrackID()!=track.MotherTrackID()) continue; // not primary
      float enedep = track.Start().E() - track.End().E();
      if(track.PdgCode()==2212){
	PrimaryPEnergy.push_back(enedep);
      }
      else if(track.PdgCode()==13 || track.PdgCode()==-13){
	PrimaryMuEnergy.push_back(enedep);
      }
    }
    // fill deposited energy of primary electron
    for(auto const& shower: ev_mcshower){
      if(shower.Origin() != 1) continue; // not neutrino-induced
      if(shower.TrackID()!=shower.MotherTrackID()) continue; // not primary
      float enedep = shower.DetProfile().E();
      if(shower.PdgCode()!=11 && shower.PdgCode()!=-11) continue; //not electron
      PrimaryEEnergy.push_back(enedep);
    }
    
    /************ VERTICES ************/
    // grab true neutrino vtx
    for(auto const& roi : ev_partroi->ROIArray()){
      if(std::abs(roi.PdgCode()) == 12 || std::abs(roi.PdgCode()) == 14) {
	_tx.resize(3,0);
	_tx[0] = roi.X();
	_tx[1] = roi.Y();
	_tx[2] = roi.Z();
	auto const offset = sce->GetPosOffsets(_tx[0],_tx[1],_tx[2]);
	_scex.resize(3,0);
	_scex[0] = _tx[0] - offset[0] + 0.7;
	_scex[1] = _tx[1] + offset[1];
	_scex[2] = _tx[2] + offset[2];
	nupixel = getProjectedPixel(_tx, wire_meta, 3);
	scenupixel = getProjectedPixel(_scex, wire_meta, 3);
      }
    }
    // grab vtx w/o tagger
    for(size_t pgraph_id = 0; pgraph_id < ev_pgraph->PGraphArray().size(); ++pgraph_id) {
      auto const& pgraph = ev_pgraph->PGraphArray().at(pgraph_id);
      _x[0] = pgraph.ParticleArray().front().X();
      _x[1] = pgraph.ParticleArray().front().Y();
      _x[2] = pgraph.ParticleArray().front().Z();
      pixel = getProjectedPixel(_x, wire_meta, 3);
      floatpixel.clear();
      for(int kk=0; kk<pixel.size(); kk++) floatpixel.push_back(pixel[kk]);

      if(std::sqrt(std::pow(_x[0]-_scex[0],2)+std::pow(_x[1]-_scex[1],2)+std::pow(_x[2]-_scex[2],2))<=5.0){
	good_vtx.push_back(_x);
	good_vtx_pixel.push_back(floatpixel);
      }
      else{bad_vtx.push_back(_x);
	bad_vtx_pixel.push_back(floatpixel);
      }
    }
    
    _x[0]=_x[1]=_x[2] = 0.;
    // grab vtx w/ old tagger
    for(size_t pgraph_id = 0; pgraph_id < ev_pgraph2->PGraphArray().size(); ++pgraph_id) {
      auto const& pgraph = ev_pgraph2->PGraphArray().at(pgraph_id);
      _x[0] = pgraph.ParticleArray().front().X();
      _x[1] = pgraph.ParticleArray().front().Y();
      _x[2] = pgraph.ParticleArray().front().Z();
      pixel = getProjectedPixel(_x, wire_meta, 3);
      floatpixel.clear();
      for(int kk=0; kk<pixel.size(); kk++) floatpixel.push_back(pixel[kk]);

      if(std::sqrt(std::pow(_x[0]-_scex[0],2)+std::pow(_x[1]-_scex[1],2)+std::pow(_x[2]-_scex[2],2))<=5.0){
	good_vtx_tag.push_back(_x);
	good_vtx_tag_pixel.push_back(floatpixel);
      }else{
	bad_vtx_tag.push_back(_x);
	bad_vtx_tag_pixel.push_back(floatpixel);
      }
    }
    
    /********* FILL ADC ********/
    // fill in adc
    for(int ip=0; ip<3; ip++){
      for(int row=0; row<wire_meta.rows(); row++){
	for(int col=0; col<wire_meta.cols(); col++){
	  float pixel=wire_img.at(ip).pixel(row,col);
	  if(pixel>=10.) hwoTagger[ip]->SetBinContent(col+1,row+1,pixel);
	}
      }
    }
    
    /******* CLUSTERS *****/
    // last cluster is a collection of unclustered hits
    for(unsigned int i=0; i<ev_cluster->size(); i++){
      numhits[i] = ev_cluster->at(i).size();

      int early=0; //before beam window
      int late=0; //after beam window
      int dwall_frac=0;

      std::vector<std::pair<float,int> > position;
      //std::vector<pos> position;

      // loop over larflow3dhits in cluster
      for(auto& hit : ev_cluster->at(i)){
      	int tick = hit.tick; //time tick
      	int wire = hit.srcwire; // Y wire

      	//are we in time with the beam
      	if( tick < beam_low ) early++;
      	if( tick > beam_high ) late++;

	//fill pos
	float dWall = calc_dWall(hit[0],hit[1],hit[2]);
	position.push_back(std::make_pair(dWall,i));
	if(dWall < dwall_thresh) dwall_frac++;

      	if ( wire_meta.min_x()<=wire && wire<wire_meta.max_x()
      	    && wire_meta.min_y()<=tick && tick<wire_meta.max_y() ) {
      	  int col = wire_meta.col( wire );
      	  int row = wire_meta.row( tick );
      	  if(binarized_instance.pixel(row,col)>0.) nufrac[i]++;
      	}
      	if ( track_meta.min_x()<=wire && wire<track_meta.max_x()
      	     && track_meta.min_y()<=tick && tick<track_meta.max_y() ) {
      	  int col = track_meta.col( wire );
      	  int row = track_meta.row( tick );

      	  float trackscore  = track_img.pixel(row,col);
          if (trackscore > ssnet_track_score_tresh){trackfrac[i]++;}
      	  float showerscore = shower_img.pixel(row,col);
          if (showerscore > ssnet_shower_score_tresh){showerfrac[i]++;}
      	  float bkgscore = 1. - trackscore - showerscore;
          if (bkgscore > ssnet_bkgd_score_tresh){bkgdfrac[i]++;}
      	}
         else{outsidefrac[i]++;}

	
      }//End of Hits Loop
      nufrac[i] /= numhits[i]; // nu hits/all hits

      //calculating ssnet fractions
      showerfrac[i] /= numhits[i];
      trackfrac[i] /= numhits[i];
      outsidefrac[i] /= numhits[i];
      bkgdfrac[i] /= numhits[i];

      //arrange the cluster by dWall
      sort(position.begin(),position.end());

     if (i<ev_cluster->size()-1){
       out_of_time[i] = is_clust_out_bounds(early,late,100);
       outside_flag[i] = is_cluster_outsidefrac_cut(outsidefrac[i],outside_frac_cut);
       shower_flag[i] = is_cluster_showerfrac_cut(showerfrac[i],shower_frac_cut);
       bkgd_flag[i] = is_cluster_bkgdfrac_cut(bkgdfrac[i],bkgd_frac_cut);
       if(position.size()>0 && position[0].first< dwall_thresh && dwall_frac>20) is_crossmu[i] =1;

       //loop over all hits and fill histograms
       for(auto& hit : ev_cluster->at(i)){
	 int tick = hit.tick; //time tick
	 int wire = hit.srcwire; // Y wire
	 if ( wire_meta.min_x()<=wire && wire<wire_meta.max_x()
	      && wire_meta.min_y()<=tick && tick<wire_meta.max_y() ) {
	   int col = wire_meta.col( wire );
	   int row = wire_meta.row( tick );
	   hclusterid->SetBinContent(col+1,row+1,i);
	   hnufrac->SetBinContent(col+1,row+1,nufrac[i]);
	   hcroi->SetBinContent(col+1,row+1,outsidefrac[i]);
	   hssnetbg->SetBinContent(col+1,row+1,bkgdfrac[i]);
	   hssnetshower->SetBinContent(col+1,row+1,showerfrac[i]);
	   hssnettrack->SetBinContent(col+1,row+1,trackfrac[i]);
	   hearly->SetBinContent(col+1,row+1,early);
	   hlate->SetBinContent(col+1,row+1,late);
	   hdwall->SetBinContent(col+1,row+1,dwall_frac);
	 }
       }

     }
     else{
       out_of_time[i] = 0;
       outside_flag[i] = 0;
       shower_flag[i] = 0;
       bkgd_flag[i] = 0;
       
       //loop over all hits and fill histograms
       for(auto& hit : ev_cluster->at(i)){
	 int tick = hit.tick; //time tick
	 int wire = hit.srcwire; // Y wire
	 if ( wire_meta.min_x()<=wire && wire<wire_meta.max_x()
	      && wire_meta.min_y()<=tick && tick<wire_meta.max_y() ) {
	   int col = wire_meta.col( wire );
	   int row = wire_meta.row( tick );
	   hclusterid->SetBinContent(col+1,row+1,-1);
	   hnufrac->SetBinContent(col+1,row+1,nufrac[i]);
	   hcroi->SetBinContent(col+1,row+1,-1);
	   hssnetbg->SetBinContent(col+1,row+1,-1);
	   hssnetshower->SetBinContent(col+1,row+1,-1);
	   hssnettrack->SetBinContent(col+1,row+1,-1);
	   hearly->SetBinContent(col+1,row+1,-1);
	   hlate->SetBinContent(col+1,row+1,-1);
	   hdwall->SetBinContent(col+1,row+1,-1);
	 }
       }
       
     }
     
     // efficiency 
     if(doeff){
       if ( (out_of_time[i] != 0) || (outside_flag[i] != 0) ||( bkgd_flag[i] != 0 ) || (is_crossmu[i] !=-1) ){
	 // loop over larflow3dhits in cluster
	 for(auto& hit : ev_cluster->at(i)){
	   int tick = hit.tick; //time tick
	   int wire = hit.srcwire; // Y wire
	   if ( wire_meta.min_x()<=wire && wire<wire_meta.max_x()
		&& wire_meta.min_y()<=tick && tick<wire_meta.max_y() ) {
	     int col = wire_meta.col( wire );
	     int row = wire_meta.row( tick );
	     if(image_hist1->GetBinContent(row+1,col+1)==2) final_sum_nu_pix--;
	     else if(image_hist1->GetBinContent(row+1,col+1)==1) final_sum_cosm_pix--;
	     image_hist1->SetBinContent(row+1,col+1,0);	     
	   }
	 }
       }
     }
     // fill save_idx
     if( out_of_time[i]==0 && outside_flag[i]==0 && bkgd_flag[i]==0 && is_crossmu[i]==-1){
       save_idx.push_back(i);
     }

    }//end of cluster loop

    if(saveana && doeff){
      fout->cd();
      image_hist0->Write();
      image_hist1->Write();
    }
    //Lets get some figures of merit:
    gStyle->SetPalette(107);

    if(doeff){
      float neutrino_efficiency = final_sum_nu_pix/orig_sum_nu_pix;
      float cosmic_rejection = 1 - (final_sum_cosm_pix/orig_sum_cosm_pix);
      eff_hist->Fill(neutrino_efficiency,cosmic_rejection);
    }


    /*
    for ( int jj=0; jj<save_idx.size(); jj++ ){
      evout_cluster->emplace_back( std::move(ev_cluster->at( save_idx[jj] ) ) );
    }
    out_larlite.set_id( run, subrun, event );
    out_larlite.next_event();
    */
  }//end of entry loop

  if(doeff){
    TCanvas can2("can", "histograms ", 800, 800);
    can2.cd();
    eff_hist->Draw();    
    can2.SaveAs("Eff_Rej.root");
    gStyle->SetStatY(0.9);
    gStyle->SetStatX(0.30);

    TH1D* rej_hist = eff_hist->ProjectionY();
    rej_hist->Draw();
    can2.SaveAs("Rej_1d.root");

    TH1D* keep_hist = eff_hist->ProjectionX();
    keep_hist->Draw();
    can2.SaveAs("Eff_1d.root");
  }
  
  if(saveana){
    fout->cd();
    tree->Write();
    fout->Close();
  }
  

  io_supera.finalize();
  io_mcinfo.close();
  io_ubclust.close();
  io_ubmrcnn.finalize();
  io_larcvtruth.finalize();
  io_ssnet.finalize();
  //out_larlite.close();
  out_larcv.finalize();
  
  return 0;
}

int is_clust_out_bounds(int early_count, int late_count, int threshold){
  /*
    This function takes in a count of how many hits are early and late.
    There is an optional argument for the threshold of how many hits too early or
    late is too many. Defaulted to 0. Funtion returns:
    1 if too many hits are early
    2 if too many hits are late
    1 if both too early and too late
    0 otherwise
  */
  int result =0;
  if (early_count > threshold) {result =1;}
  else if (late_count > threshold) {result =2;}
  else if (late_count+early_count > threshold) { result =1;}
  return result;
}

int is_cluster_outsidefrac_cut(float outsidefrac, float threshold){
  //returns whether outnow cluster passes fraction outside croi cout
  // 1 if fails
  // 0 if passes
  int result = 0;
  if (outsidefrac > threshold){result = 1;}
  return result;
}

int is_cluster_showerfrac_cut(float showerfrac, float threshold){
  //returns whether outnow cluster passes shower fraction cout
  // 1 if fails
  // 0 if passes
  int result = 0;
  if (showerfrac > threshold){result = 1;}
  return result;
}

int is_cluster_bkgdfrac_cut(float bkgdfrac, float threshold){
  //returns whether outnow cluster passes background fraction cout
  // 1 if fails
  // 0 if passes
  int result = 0;
  if (bkgdfrac > threshold){result = 1;}
  return result;
}

float calc_dWall(float x, float y, float z){

  const ::larutil::Geometry* geo = ::larutil::Geometry::GetME();
  const float xmin = 0.;
  const float xmax = geo->DetHalfWidth()*2.;
  const float ymin = -1.*geo->DetHalfHeight();
  const float ymax = geo->DetHalfHeight();
  const float zmin = 0.;
  const float zmax = geo->DetLength();

  float mindist = 1.0e4;

  if(std::abs(y - ymin)< mindist) mindist = std::abs(y - ymin);
  if(std::abs(y - ymax)< mindist) mindist = std::abs(y - ymax);
  if(std::abs(z - zmin)< mindist) mindist = std::abs(z - zmin);
  if(std::abs(z - zmax)< mindist) mindist = std::abs(z - zmax);

  return mindist;

}

std::vector<int> getProjectedPixel( const std::vector<float>& pos3d,
				    const larcv::ImageMeta& meta,
				    const int nplanes,
				    const float fracpixborder ) {
  std::vector<int> img_coords( nplanes+1, -1 );
  float row_border = fabs(fracpixborder)*meta.pixel_height();
  float col_border = fabs(fracpixborder)*meta.pixel_width();

  // tick/row
  float tick = pos3d[0]/(::larutil::LArProperties::GetME()->DriftVelocity()*::larutil::DetectorProperties::GetME()->SamplingRate()*1.0e-3) + 3200.0;
  if ( tick<meta.min_y() ) {
    if ( tick>meta.min_y()-row_border )
      // below min_y-border, out of image
      img_coords[0] = meta.rows()-1; // note that tick axis and row indicies are in inverse order (same order in larcv2)
    else
      // outside of image and border
      img_coords[0] = -1;
  }
  else if ( tick>meta.max_y() ) {
    if ( tick<meta.max_y()+row_border )
      // within upper border
      img_coords[0] = 0;
    else
      // outside of image and border
      img_coords[0] = -1;
  }
  else {
    // within the image
    img_coords[0] = meta.row( tick );
  }

  // Columns
  Double_t xyz[3] = { pos3d[0], pos3d[1], pos3d[2] };
  // there is a corner where the V plane wire number causes an error
  if ( (pos3d[1]>-117.0 && pos3d[1]<-116.0) && pos3d[2]<2.0 ) {
    xyz[1] = -116.0;
  }
  for (int p=0; p<nplanes; p++) {
    float wire = larutil::Geometry::GetME()->WireCoordinate( xyz, p );

    // get image coordinates
    if ( wire<meta.min_x() ) {
      if ( wire>meta.min_x()-col_border ) {
	// within lower border
	img_coords[p+1] = 0;
      }
      else
	img_coords[p+1] = -1;
    }
    else if ( wire>=meta.max_x() ) {
      if ( wire<meta.max_x()+col_border ) {
	// within border
	img_coords[p+1] = meta.cols()-1;
      }
      else
	// outside border
	img_coords[p+1] = -1;
    }
    else
      // inside image
      img_coords[p+1] = meta.col( wire );
  }//end of plane loop

  // there is a corner where the V plane wire number causes an error
  if ( pos3d[1]<-116.3 && pos3d[2]<2.0 && img_coords[1+1]==-1 ) {
    img_coords[1+1] = 0;
  }
  return img_coords;
}
