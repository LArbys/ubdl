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

  //std::string output_hist = "outhist.root";
  //TFile fout( output_hist.c_str(), "recreate" );
  //TH2D* hancestor;

  //this should be defined with proper constants
  // larutil::LArProperties larproperties;
  // larutil::Geometry geometry;
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
  float ssnet_bkgd_score_tresh = 0.5;
  float ssnet_shower_score_tresh = 0.5;
  float ssnet_track_score_tresh = 0.5;

  //cluster background fraction to cut on (placeholder for now)
  float bkgd_frac_cut = 0.0;
  //cluster shower fraction to cut on (placeholder for now)
  float shower_frac_cut = 0.0;
  //cluster frac outside cROI to cut on (placeholder for now)
  float outside_frac_cut = 0.5;

  // dist to detector boundary threshold in cm
  const float dwall_thresh = 10.0;
  
  int nentries = io_ubpost_larcv.get_n_entries();
  TH2F eff_hist("Neutrino and Cosmic Efficiency", "Nu Efficiency, Cosmic Rejection, Pixelwise", 20,0.0,1.001,20,0.0,1.001);
  TH1D rej_hist("Cosmic Rejection", "Cosmic Rejection, Pixelwise (Nu Eff > 90%)", 25,0.0,1.001);
  eff_hist.SetOption("COLZ");
  eff_hist.SetXTitle("Neutrino Efficiency");
  eff_hist.SetYTitle("Cosmic Rejection");
  // rej_hist.SetOption("COLZ");
  rej_hist.SetXTitle("Cosmic Rejection");

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


    //grab Y plane wire image
    auto const& wire_img = ev_img->Image2DArray().at( 2 );
    auto const& wire_meta = wire_img.meta();

    //grab Y plane ancestor image
    auto const& anc_img = ev_ancestor->Image2DArray().at( 2 );
    auto const& anc_meta = anc_img.meta();

    //grab Y plane instance image
    auto const& inst_img = ev_instance->Image2DArray().at( 2 );
    auto const& inst_meta = inst_img.meta();

    //grab Y plane SSNet images
    auto const& track_img = ev_ssnet->Image2DArray().at(0);
    auto const& shower_img = ev_ssnet->Image2DArray().at(1);
    auto const& track_meta = track_img.meta();

    //char histname_event[100];
    //sprintf(histname_event,"hancestor_run%d_event%d",run,event);
    //hancestor = new TH2D(histname_event,"",meta.cols(), 0, meta.cols(), meta.rows(),0,meta.rows());

    
    // loop over all clusters
    // last one is a collection of unclustered hits
    for(unsigned int i=0; i<ev_cluster->size(); i++){
      numhits[i] = ev_cluster->at(i).size();

      int early=0; //before beam window
      int late=0; //after beam window

      std::vector<std::pair<float,int> > position;
      //std::vector<pos> poaition;
      
      // loop over larflow3dhits in cluster
      for(auto& hit : ev_cluster->at(i)){
      	int tick = hit.tick; //time tick
      	int wire = hit.srcwire; // Y wire

      	//are we in time with the beam
      	if( tick < beam_low ) early++;
      	if( tick > beam_high ) late++;

      	if ( anc_meta.min_x()<=wire && wire<anc_meta.max_x()
      	    && anc_meta.min_y()<=tick && tick<anc_meta.max_y() ) {
      	  int col = anc_meta.col( wire );
      	  int row = anc_meta.row( tick );
      	  float nuscore  = anc_img.pixel(row,col);
      	  if(nuscore>-1) nufrac[i]++;
      	  //hancestor->SetBinContent(col+1, row+1, nuscore);
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

        //fill pos
	float dWall = calc_dWall(hit[0],hit[1],hit[2]);
	position.push_back(std::make_pair(dWall,i));
	
	
      }//End of Hits Loop
      nufrac[i] /= numhits[i]; // nu hits/all hits


      //calculating ssnet fractions
      showerfrac[i] /= numhits[i];
      trackfrac[i] /= numhits[i];
      outsidefrac[i] /= numhits[i];
      bkgdfrac[i] /= numhits[i];

      //flags for ssnet (not using yet)
      // int bkgdflag = is_cluster_bkgdfrac_cut(bkgdfrac[i],bkgd_frac_cut);
      // int showerflag = is_cluster_showerfrac_cut(showerfrac[i],shower_frac_cut);

      //flag cross mu & through mu
      //first we arrange the cluster by dWall
      sort(position.begin(),position.end());
      if(position.size()>0 && position[0].first< dwall_thresh) is_crossmu[i] =1; //this is actually true for both crossing and through-going muons, will refine 
      
     if (i<ev_cluster->size()-1){
       out_of_time[i] = is_clust_out_bounds(early,late,100);
       outside_flag[i] = is_cluster_outsidefrac_cut(outsidefrac[i],outside_frac_cut);
     }
     else{
       out_of_time[i] = 0;
       outside_flag[i] = 0;
     }
    }//end of cluster loop

    //fout.cd();
    //hancestor->Write();

    //Lets get some figures of merit:
    // TCanvas can("can", "histograms ", 2400, 800);
    // can.cd();
    gStyle->SetPalette(107);
    TH2F image_hist("Y-Plane, Updated", "Y-Plane Everything", 1008, 0.0, 1008.0, 3456,0.0,3456);
    image_hist.SetOption("COLZ");
    larcv::Image2D binarized_adc = wire_img;
    binarized_adc.binary_threshold(10.0, 0.0, 1.0);
    larcv::Image2D binarized_instance =  inst_img;
    binarized_instance.binary_threshold(-1.0, 0.0, 1.0);
    binarized_instance.eltwise(binarized_adc);
    larcv::Image2D neut_or_cosmic = binarized_adc;

    auto const& adcmeta = binarized_adc.meta();
    for (int r=0;r<adcmeta.rows();r++){
      for (int c=0;c<adcmeta.cols();c++){
        if (binarized_instance.pixel(r,c) == 1){
          neut_or_cosmic.set_pixel(r,c,2);
        }
        image_hist.SetBinContent(r,c, neut_or_cosmic.pixel(r,c));
      }
    }
    // image_hist.Draw();
    // image_hist.Write();
    std::string str = "Neut_or_Cosm_All_"+std::to_string(i)+".png";
    char cstr[str.size() + 1];
    str.copy(cstr,str.size()+1);
    cstr[str.size()] = '\0';
    // can.SaveAs(cstr);
    image_hist.SetTitle("Y-Plane Cuts");
    for(unsigned int i=0; i<ev_cluster->size(); i++){
      if ((out_of_time[i] != 0) || (outside_flag[i] !=0) ){
        // loop over larflow3dhits in cluster
        for(auto& hit : ev_cluster->at(i)){
          int tick = hit.tick; //time tick
          int wire = hit.srcwire; // Y wire
          if ( neut_or_cosmic.meta().min_x()<=wire && wire<neut_or_cosmic.meta().max_x()
               && neut_or_cosmic.meta().min_y()<=tick && tick<neut_or_cosmic.meta().max_y() ) {
            int col = neut_or_cosmic.meta().col( wire );
            int row = neut_or_cosmic.meta().row( tick );
            neut_or_cosmic.set_pixel(row,col,0);
            image_hist.SetBinContent(row,col,0);
          }
        }

      }

    }
    float orig_sum_nu_pix = 0;
    float final_sum_nu_pix = 0;
    float orig_sum_cosm_pix = 0;
    float final_sum_cosm_pix = 0;
    for (int r=0;r<1008;r++){
      for (int c=0;c<3456;c++){
        if (binarized_instance.pixel(r,c) == 1){
          orig_sum_nu_pix++;
        }
        else if(binarized_adc.pixel(r,c) ==1){
          orig_sum_cosm_pix++;
        }
        if (image_hist.GetBinContent(r,c) ==2){
          final_sum_nu_pix++;
        }
        else if (image_hist.GetBinContent(r,c) ==1){
          final_sum_cosm_pix++;
        }
      }
    }


    float neutrino_efficiency = final_sum_nu_pix/orig_sum_nu_pix;
    float cosmic_rejection = 1 - (final_sum_cosm_pix/orig_sum_cosm_pix);
    std::cout << orig_sum_nu_pix << " Orig Count Nu" << std::endl;
    std::cout << final_sum_nu_pix << " Final Count Nu" << std::endl;
    std::cout << neutrino_efficiency << " Fract " << std::endl;

    std::cout << orig_sum_cosm_pix << " Orig Count Cosm" << std::endl;
    std::cout << final_sum_cosm_pix << " Final Count Cosm" << std::endl;
    std::cout << cosmic_rejection << " Cosmic Rejection" << std::endl;

    eff_hist.Fill(neutrino_efficiency,cosmic_rejection);
    if (neutrino_efficiency >= 0.9){
      rej_hist.Fill(cosmic_rejection);
    }

    // image_hist.Draw();
    // image_hist.Write();
    str = "Neut_or_Cosm_Removed_"+std::to_string(i)+".png";
    cstr[str.size() + 1];
    str.copy(cstr,str.size()+1);
    cstr[str.size()] = '\0';
    // can.SaveAs(cstr);
    image_hist.Reset();


    // std::cout << count_neut << " Neutrino Pixels" << std::endl;
    // std::cout << count_neut_thresh << " Neutrino Pixels when adc thresholded" << std::endl;
    // std::cout << count_neut_thresh/count_neut << " Fraction kept by that" << std::endl;



  }//end of entry loop
  TCanvas can2("can", "histograms ", 800, 800);
  can2.cd();
  eff_hist.Draw();
  // eff_hist.Write();
  can2.SaveAs("Eff_Rej_outside.png");
  gStyle->SetStatY(0.9);
  gStyle->SetStatX(0.30);
  rej_hist.Draw();
  // rej_hist.Write();

  // can2.SaveAs("Eff_Rej_outside.png");
  can2.SaveAs("Rej_1d.png");

  io_ubpost_larcv.finalize();
  io_ubpost_larlite.close();
  io_ubclust.close();
  io_ubmrcnn.finalize();
  io_larcvtruth.finalize();
  io_ssnet.finalize();

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
