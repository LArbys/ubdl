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
//#include "larcv/core/DataFormat/EventClusterMask.h"

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

  larcv::IOManager io_ancestor( larcv::IOManager::kREAD, "", larcv::IOManager::kTickBackward );
  io_ancestor.add_in_file( ancestor );
  io_ancestor.initialize();
  
  int nentries = io_ubpost_larcv.get_n_entries();
  for (int i=0; i<nentries; i++) {
    io_ubpost_larcv.read_entry(i);
    io_ubpost_larlite.go_to(i);
    io_ubclust.go_to(i);
    io_ubmrcnn.read_entry(i);
    io_ancestor.read_entry(i); //this needs to change? not sure RSE is same
    
    std::cout << "entry " << i << std::endl;

    // in
    auto ev_img            = (larcv::EventImage2D*)io_ubpost_larcv.get_data( larcv::kProductImage2D, "wire" );
    auto ev_chstatus       = (larcv::EventChStatus*)io_ubpost_larcv.get_data( larcv::kProductChStatus,"wire");

    auto ev_cluster        = (larlite::event_larflowcluster*)io_ubpost_larlite.get_data(larlite::data::kLArFlowCluster,  "rawmrcnn" );

    auto ev_mctrack        = (larlite::event_mctrack*)io_ubpost_larlite.get_data(larlite::data::kMCTrack,  "mcreco" );
    auto ev_mcshower       = (larlite::event_mcshower*)io_ubpost_larlite.get_data(larlite::data::kMCShower, "mcreco" );
    
    auto ev_opflash_beam   = (larlite::event_opflash*)io_ubpost_larlite.get_data(larlite::data::kOpFlash, "simpleFlashBeam" );
    auto ev_opflash_cosmic = (larlite::event_opflash*)io_ubpost_larlite.get_data(larlite::data::kOpFlash, "simpleFlashCosmic" );

    auto ev_ancestor       = (larcv::EventImage2D*)io_ancestor.get_data( larcv::kProductImage2D, "ancestor" ); //ancestor
    auto ev_ssnet          = (larcv::EventImage2D*)io_ancestor.get_data( larcv::kProductImage2D, "segment" ); //ssnet

    //auto ev_clustermask    = (larcv::EventClusterMask*)io_ubmrcnn.get_data( larcv::kClusterMask, "mrcnn_masks");

    
    run    = io_ubpost_larcv.event_id().run();
    subrun = io_ubpost_larcv.event_id().subrun();
    event  = io_ubpost_larcv.event_id().event();
    std::cout <<"run "<< run <<" subrun " << subrun <<" event " << event << std::endl;

    // some flags for clusters
    std::vector<int> is_thrumu(ev_cluster->size(),-1); //is cluster crossing two TPC faces?
    std::vector<int> is_crossmu(ev_cluster->size(),-1); //is cluster crossing one TPC face?
    std::vector<int> out_of_time(ev_cluster->size(),-1); //is cluster partially outside of beam window?
    std::vector<float> nufrac(ev_cluster->size(),-1); //how many cluster pixels are neutrino?
    
    // loop over all clusters
    // last one is a collection of unclustered hits
    for(unsigned int i=0; i<ev_cluster->size(); i++){
      // loop over larflow3dhits in cluster
      for(auto& hit : ev_cluster->at(i)){
	int tick = hit.tick; //time tick in full img coord (I think)
	int wire = hit.srcwire; // Y wire in full img coord (I think)

	/* do stuff here */
      }
    }
    
  }//end of entry loop
  
  io_ubpost_larcv.finalize();
  io_ubpost_larlite.close();
  io_ubclust.close();
  io_ubmrcnn.finalize();
  io_ancestor.finalize();
  
  return 0;
}
