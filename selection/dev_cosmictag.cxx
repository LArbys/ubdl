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
#include "LArUtil/LArProperties.h"
#include "LArUtil/Geometry.h"
#include "LArUtil/ClockConstants.h"

// larcv
#include "larcv/core/DataFormat/IOManager.h"
#include "larcv/core/DataFormat/EventImage2D.h"
#include "larcv/core/DataFormat/EventChStatus.h"
#include "larcv/core/DataFormat/EventClusterMask.h"

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

  std::string ubpost_larcv  = argv[1];
  std::string ubpost_larlite = argv[2];
  std::string ubmrcnn = argv[3];
  std::string ubclust = argv[4];
  std::string ancestor = argv[5];
  std::string ssnet = argv[6];
  //std::string infill = argv[7];
  std::string outfile_larlite = argv[7];

  std::string saveme = "";
  std::string calc_eff="";
  if(nargs>8){
    saveme = argv[8];
    calc_eff = argv[8];
    if(nargs>9) saveme = argv[9];
  }
  bool savehist=false;
  if(saveme=="save") savehist=true; //only if we give "save" as last argument save histograms
  bool doeff=false;
  if(calc_eff=="eff") doeff=true; 
  
  int run, subrun, event;

  // input
  larcv::IOManager io_ubpost_larcv( larcv::IOManager::kREAD, "", larcv::IOManager::kTickForward );
  io_ubpost_larcv.add_in_file( ubpost_larcv );
  io_ubpost_larcv.initialize();

  larlite::storage_manager io_ubpost_larlite( larlite::storage_manager::kREAD );
  io_ubpost_larlite.add_in_filename( ubpost_larlite );
  io_ubpost_larlite.open();

  larlite::storage_manager io_ubclust( larlite::storage_manager::kREAD );
  io_ubclust.add_in_filename( ubclust );
  io_ubclust.open();

  larcv::IOManager io_ubmrcnn( larcv::IOManager::kREAD, "", larcv::IOManager::kTickBackward );
  io_ubmrcnn.add_in_file( ubmrcnn );
  io_ubmrcnn.initialize();

  larcv::IOManager io_larcvtruth( larcv::IOManager::kREAD, "", larcv::IOManager::kTickBackward );
  io_larcvtruth.add_in_file( ancestor );
  io_larcvtruth.initialize();

  larcv::IOManager io_ssnet( larcv::IOManager::kREAD, "", larcv::IOManager::kTickBackward );
  io_ssnet.add_in_file( ssnet );
  io_ssnet.initialize();

  // output
  //std::string outfile_larlite = "out_larflowclust.root"; //temp
  larlite::storage_manager out_larlite( larlite::storage_manager::kWRITE );
  out_larlite.set_out_filename( outfile_larlite );
  out_larlite.open();

  std::string output_hist = "outhist.root";
  TFile* fout;
  if(savehist) fout = new TFile( output_hist.c_str(), "recreate" );

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

  //cluster background fraction to cut on (placeholder for now)
  const float bkgd_frac_cut = 0.3;
  //cluster shower fraction to cut on (placeholder for now)
  const float shower_frac_cut = 0.1;
  //cluster frac outside cROI to cut on (placeholder for now)
  const float outside_frac_cut = 0.1;

  // dist to detector boundary threshold in cm
  const float dwall_thresh = 10.0;

  //nu fraction for cluster neutrino labeling
  const float nufrac_thresh = 0.5;

  TH2F* eff_hist = NULL;
  if(doeff){
    eff_hist = new TH2F("eff_hist", "Nu Efficiency, Cosmic Rejection, Pixelwise", 20,0.0,1.001,20,0.0,1.001);
    eff_hist->SetOption("COLZ");
    eff_hist->SetXTitle("Neutrino Efficiency");
    eff_hist->SetYTitle("Cosmic Rejection");
  }
  //
  TH1D* hnu = new TH1D("hdWall_nu","",128.,0,128.);
  TH1D* hcr = new TH1D("hdWall_cr","",128.,0,128.);
  TH1D* hnuPointsOut = new TH1D("hPointsOut_nu","",200,0,400.);
  TH1D* hcrPointsOut = new TH1D("hPointsOut_cr","",200,0,400.);

  int nentries = io_ubpost_larcv.get_n_entries();
  nentries = 10; //temp
  for (int i=0; i<nentries; i++) {

    io_ubpost_larcv.read_entry(i);
    io_ubpost_larlite.go_to(i);
    io_ubclust.go_to(i);
    io_ubmrcnn.read_entry(i);
    io_larcvtruth.read_entry(i);
    io_ssnet.read_entry(i);

    std::cout << "entry " << i << std::endl;

    // in
    auto ev_img            = (larcv::EventImage2D*)io_ubpost_larcv.get_data( larcv::kProductImage2D, "wire" );
    auto ev_chstatus       = (larcv::EventChStatus*)io_ubpost_larcv.get_data( larcv::kProductChStatus,"wire");

    auto ev_cluster        = (larlite::event_larflowcluster*)io_ubclust.get_data(larlite::data::kLArFlowCluster,  "rawmrcnn" );

    auto ev_mctrack        = (larlite::event_mctrack*)io_ubpost_larlite.get_data(larlite::data::kMCTrack,  "mcreco" );
    auto ev_mcshower       = (larlite::event_mcshower*)io_ubpost_larlite.get_data(larlite::data::kMCShower, "mcreco" );

    auto ev_opflash_beam   = (larlite::event_opflash*)io_ubpost_larlite.get_data(larlite::data::kOpFlash, "simpleFlashBeam" );
    auto ev_opflash_cosmic = (larlite::event_opflash*)io_ubpost_larlite.get_data(larlite::data::kOpFlash, "simpleFlashCosmic" );

    auto ev_ancestor       = (larcv::EventImage2D*)io_larcvtruth.get_data( larcv::kProductImage2D, "ancestor" );
    auto ev_instance       = (larcv::EventImage2D*)io_larcvtruth.get_data( larcv::kProductImage2D, "instance" );

    auto ev_ssnet          = (larcv::EventImage2D*)io_ssnet.get_data( larcv::kProductImage2D, "uburn_plane2" );

    auto ev_clustermask    = (larcv::EventClusterMask*)io_ubmrcnn.get_data( larcv::kProductClusterMask, "mrcnn_masks");


    //out
    auto evout_cluster     = (larlite::event_larflowcluster*)out_larlite.get_data(larlite::data::kLArFlowCluster, "cosmictag");
    
    run    = io_ubpost_larcv.event_id().run();
    subrun = io_ubpost_larcv.event_id().subrun();
    event  = io_ubpost_larcv.event_id().event();
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
    auto const& wire_img = ev_img->Image2DArray().at( 2 );
    auto const& wire_meta = wire_img.meta();

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
    
    larcv::Image2D binarized_adc = wire_img;
    binarized_adc.binary_threshold(10.0, 0.0, 1.0);
    larcv::Image2D binarized_instance =  inst_img;
    binarized_instance.binary_threshold(-1.0, 0.0, 1.0);
    binarized_instance.eltwise(binarized_adc);

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
    // loop over all clusters
    // last one is a collection of unclustered hits
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
      	  if(binarized_instance.pixel(row,col)>0.){
	    nufrac[i]++;
	    hnu->Fill(dWall);}
	  else hcr->Fill(dWall);
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

      //flag cross mu & through mu
      //first we arrange the cluster by dWall
      sort(position.begin(),position.end());


     if (i<ev_cluster->size()-1){
       out_of_time[i] = is_clust_out_bounds(early,late,100);
       outside_flag[i] = is_cluster_outsidefrac_cut(outsidefrac[i],outside_frac_cut);
       shower_flag[i] = is_cluster_showerfrac_cut(showerfrac[i],shower_frac_cut);
       bkgd_flag[i] = is_cluster_bkgdfrac_cut(bkgdfrac[i],bkgd_frac_cut);
       if(position.size()>0 && position[0].first< dwall_thresh && dwall_frac>20) is_crossmu[i] =1;

       if(nufrac[i]>=nufrac_thresh) hnuPointsOut->Fill(dwall_frac);
       else hcrPointsOut->Fill(dwall_frac);
     }
     else{
       out_of_time[i] = 0;
       outside_flag[i] = 0;
       shower_flag[i] = 0;
       bkgd_flag[i] = 0;
     }

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

    if(savehist && doeff){
      fout->cd();
      image_hist0->Write();
      image_hist1->Write();
    }
    //Lets get some figures of merit:
    gStyle->SetPalette(107);

    if(doeff){
      float neutrino_efficiency = final_sum_nu_pix/orig_sum_nu_pix;
      float cosmic_rejection = 1 - (final_sum_cosm_pix/orig_sum_cosm_pix);
      std::cout << orig_sum_nu_pix << " Orig Count Nu" << std::endl;
      std::cout << final_sum_nu_pix << " Final Count Nu" << std::endl;
      std::cout << neutrino_efficiency << " Fract " << std::endl;
      
      std::cout << orig_sum_cosm_pix << " Orig Count Cosm" << std::endl;
      std::cout << final_sum_cosm_pix << " Final Count Cosm" << std::endl;
      std::cout << cosmic_rejection << " Cosmic Rejection" << std::endl;
      
      eff_hist->Fill(neutrino_efficiency,cosmic_rejection);

    }


    // fill output
    for ( int jj=0; jj<save_idx.size(); jj++ ){
      evout_cluster->emplace_back( std::move(ev_cluster->at( save_idx[jj] ) ) );
    }
    
    out_larlite.set_id( run, subrun, event );
    out_larlite.next_event();
      
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
  
  if(savehist){
    fout->cd();
    hnu->Write();
    hcr->Write();
    hnuPointsOut->Write();
    hcrPointsOut->Write();
    fout->Close();
  }
  

  io_ubpost_larcv.finalize();
  io_ubpost_larlite.close();
  io_ubclust.close();
  io_ubmrcnn.finalize();
  io_larcvtruth.finalize();
  io_ssnet.finalize();
  out_larlite.close();
  
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
