#include <iostream>

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
int main( int nargs, char** argv ) {

  gStyle->SetOptStat(0);

  std::string ubpost_larcv  = argv[1];
  std::string ubpost_larlite = argv[2];
  std::string ubmrcnn = argv[3];
  std::string ubclust = argv[4];
  std::string ancestor = argv[5];
  std::string ssnet = argv[6];
  //std::string infill = argv[6];

  int run, subrun, event;

  // input
  larcv::IOManager io_ubpost_larcv( larcv::IOManager::kREAD, "", larcv::IOManager::kTickBackward );
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
  float beam_low = 3200.;
  float beam_high = 3200 + 256./0.111/0.5; //width/drift speed / us per tick 

  //fraction of cluster hits not in time with beam
  float timewin_thresh = 0.1;
  
  int nentries = io_ubpost_larcv.get_n_entries();

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
	  float showerscore = shower_img.pixel(row,col);
	  float bkgscore = 1. - trackscore - showerscore;

	}
	
      }
      nufrac[i] /= numhits[i]; // nu hits/all hits
      
      if(early/numhits[i] >= timewin_thresh) out_of_time[i] = 1;
      if(late/numhits[i] >= timewin_thresh) out_of_time[i] = 2;
      else if(early/numhits[i]<timewin_thresh && late/numhits[i]<timewin_thresh) out_of_time[i] = 0;

	//flag cross mu & through mu
      //first we arrange the cluster by y coordinate
      
    }//end of cluster loop
    
    //fout.cd();
    //hancestor->Write();

  }//end of entry loop

  io_ubpost_larcv.finalize();
  io_ubpost_larlite.close();
  io_ubclust.close();
  io_ubmrcnn.finalize();
  io_larcvtruth.finalize();
  io_ssnet.finalize();
  
  return 0;
}
