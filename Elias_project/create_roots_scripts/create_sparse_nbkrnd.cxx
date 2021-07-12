#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include <math.h>
// ROOT
#include "TH2F.h"
#include "TCanvas.h"
#include "TStyle.h"
// larutil
#include "LArUtil/LArProperties.h"
#include "LArUtil/DetectorProperties.h"
#include "LArUtil/Geometry.h"
#include "LArUtil/ClockConstants.h"
#include "LArUtil/SpaceChargeMicroBooNE.h"
// larcv
#include "larcv/core/DataFormat/IOManager.h"
#include "larcv/core/DataFormat/EventImage2D.h"
#include "larcv/core/DataFormat/EventPGraph.h"
#include "larcv/core/DataFormat/SparseImage.h"
#include "larcv/core/DataFormat/EventSparseImage.h"
// larlite
#include "DataFormat/storage_manager.h"
#include "DataFormat/mctruth.h"



// initalize helper functions
void print_signal();
void print_rse(int run, int subrun, int event);
std::vector<int> get_truth_info(larlite::storage_manager* io_larlite, int entry, int event_wire, int subrun_wire, std::vector<int> pxl_count);



// main fuction to load in a dlana root file, make a copy of the image2d with a loop, and save to a root file
int main(int nargs, char** argv){
	std::cout << "Hello world " << "\n";
	std::cout << "Args Required, Order Matters:\n";
	std::cout << "Inputfile with Image2D\n";
	std::cout << "Output Directory\n";
	std::cout << "\n";
	std::cout << "Args Optional:\n";
	std::cout << "Specific Entry to Do (Default is all entries)\n";

  if (nargs < 3){
		std::cout << "Not Enough Args\n";
		return 1;
	}
  std::string input_file = argv[1];
	std::string output_dir = std::string(argv[2]) + "/";

  int specific_entry = -1;
	if (nargs > 3){
		specific_entry = atoi(argv[3]);
	}

	std::cout << "Running with Args:\n";
	for (int i = 0; i<nargs;i++){
		std::cout << "Arg " << i << " : " << argv[i] << "\n";
	}

  // create root output file
  std::string output_file = output_dir + "output_sparse.root";
  //create larcv IO IOManager
  // create 2 io managers- one to read one to write
	larcv::IOManager* in_larcv  = new larcv::IOManager(larcv::IOManager::kREAD,"IOManager_In", larcv::IOManager::kTickBackward);
  larcv::IOManager* out_larcv  = new larcv::IOManager(larcv::IOManager::kWRITE,"IOManager_Out", larcv::IOManager::kTickForward);
  // load in and intialize larcv products
  // reverse necessary due to how products were created
  in_larcv->reverse_all_products();
	in_larcv->add_in_file(input_file);
	in_larcv->initialize();
  out_larcv->set_out_file(output_file);
  out_larcv->initialize();

  //createlarlite storage Manager
	larlite::storage_manager* io_larlite  = new larlite::storage_manager(larlite::storage_manager::kREAD);
  // load in and intialize larlite products
	io_larlite->add_in_filename(input_file);
	io_larlite->open();


	int start_entry = 0;
	int nentries_mc_cv = in_larcv->get_n_entries();
	std::cout << "Entries in File:  " << nentries_mc_cv << "\n";

  // loop to set entry if we only want one
  if (specific_entry != -1){
		start_entry = specific_entry;
		nentries_mc_cv = start_entry+1;
	}
	
	int zero_count = 0;
    
  // loop through all the entries in the file
	for (int entry=start_entry; entry < nentries_mc_cv; entry++){


    std::cout << std::endl << "Entry : " << entry << "\n";
		in_larcv->read_entry(entry);
		std::cout << "\n";

    // load in the event display image (contains the 3 planes)
    larcv::EventImage2D* ev_in_adc_dlreco_wire  = (larcv::EventImage2D*)(in_larcv->get_data(larcv::kProductImage2D,"wire"));
	larcv::EventImage2D* ev_in_adc_dlreco_thrumu  = (larcv::EventImage2D*)(in_larcv->get_data(larcv::kProductImage2D,"thrumu"));
    // take the larcv event image and change to a vector of images (one for each plane)
    std::vector< larcv::Image2D > img_2d_in_v_wire = ev_in_adc_dlreco_wire->Image2DArray();
	std::vector< larcv::Image2D > img_2d_in_v_thrumu = ev_in_adc_dlreco_thrumu->Image2DArray();
    // we also get the meta. We will need a copy of this to save
    const std::vector<larcv::ImageMeta> meta_orig_v = {img_2d_in_v_wire.at(0).meta(), img_2d_in_v_wire.at(1).meta(), img_2d_in_v_wire.at(2).meta() };

    // get identifying info of entry
    int run_wire = ev_in_adc_dlreco_wire->run();
		int subrun_wire = ev_in_adc_dlreco_wire->subrun();
		int event_wire = ev_in_adc_dlreco_wire->event();
		print_rse(run_wire, subrun_wire, event_wire);
	int run_thrumu = ev_in_adc_dlreco_thrumu->run();
		int subrun_thrumu = ev_in_adc_dlreco_thrumu->subrun();
		int event_thrumu = ev_in_adc_dlreco_thrumu->event();
        if (run_wire != run_thrumu || subrun_wire != subrun_thrumu || event_wire != event_thrumu){
            std::cout << "PROBLEM: RSE does not match\n";
            std::cout << "Wire: ";
            print_rse(run_wire, subrun_wire, event_wire);
            std::cout << "thrumu: ";
            print_rse(run_thrumu, subrun_thrumu, event_thrumu);
        }

	
       
       
    // now create the Image2D, will be used to initialize the SparseImage
    std::vector<larcv::Image2D> out_v;
	std::vector<int> pxl_count;
	pxl_count.resize(3);

    // loop through all the pixels of the input to get the pixel information (number is standard uboone image size)
    for (int plane = 0; plane<3;plane++){
      // column = time tic, row = wire number
      // create single image2d for the plane
      larcv::Image2D single_out( meta_orig_v[plane] );
      single_out.paint(0.0);
      for (int col = 0; col<3456;col++){
        for (int row = 0; row<1008;row++){
          // get the pixel value for each plane
          double val = img_2d_in_v_wire[plane].pixel(row,col);
		  if (img_2d_in_v_thrumu[plane].pixel(row,col) != 0) val = 0;
		  if (val != 0) pxl_count[plane]++;
          // now save to new image2d
          single_out.set_pixel(row,col,val);
        }//end of loop over rows
      }//end of loop over cols
      out_v.emplace_back( std::move(single_out) );
    }//end of loop over planes
	
	// Getting truth information
	std::vector<int> truth = get_truth_info(io_larlite, entry, event_wire, subrun_wire, pxl_count);
	int event = truth[0];
	int subrun = truth[1];
	int good_entry = truth[2]; // 1 is good, 0 is bad
	
	if (good_entry == 1){
		// Saving sparse:
		std::vector<float> thresholds;
		thresholds.push_back(10);
		
		larcv::EventSparseImage* ev_out_adc_dlreco_sparse = (larcv::EventSparseImage*) out_larcv->get_data(larcv::kProductSparseImage,"nbkrnd_sparse");
		// ev_out_adc_dlreco_sparse->clear();
		larcv::SparseImage out_sparse(out_v, thresholds);
		std::cout << "out_sparse length: " << out_sparse.len() << "\n";
		if (out_sparse.len() == 0) zero_count++;
		// ^note that this^ will not necessarily catch any zero planes
		ev_out_adc_dlreco_sparse->Emplace( std::move(out_sparse) );
		print_rse(run_wire, subrun, event);
		out_larcv->set_id( run_wire, subrun, event);
	    out_larcv->save_entry();
	}
	
	} //End of entry loop
	
	std::cout << "total zero count: " << zero_count << "\n";
		
  // close larcv I) manager
	in_larcv->finalize();
	delete in_larcv;
  out_larcv->reverse_all_products();
  out_larcv->finalize();
  delete out_larcv;
  io_larlite->close();
  delete io_larlite;

  print_signal();
  return 0;
	}//End of main


void print_signal(){
	std::cout << "\n\n";
	std::cout << "/////////////////////////////////////////////////////////////\n";
	std::cout << "/////////////////////////////////////////////////////////////\n";
	std::cout << "\n\n";
  std::cout << "         _==/            i     i           \\==_ \n";
	std::cout << "        /XX/             |\\___/|            \\XX\\    \n";
	std::cout << "       /XXXX\\            |XXXXX|            /XXXX\\   \n";
	std::cout << "      |XXXXXX\\_         _XXXXXXX_         _/XXXXXX|   \n";
	std::cout << "     XXXXXXXXXXXxxxxxxxXXXXXXXXXXXxxxxxxxXXXXXXXXXXX   \n";
	std::cout << "    |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX|  \n";
	std::cout << "    XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX   \n";
	std::cout << "    |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX|   \n";
	std::cout << "     XXXXXX/^^^^^\\XXXXXXXXXXXXXXXXXXXXX/^^^^^\\XXXXXX    \n";
	std::cout << "      |XXX|       \\XXX/^^\\XXXXX/^^\\XXX/       |XXX|    \n";
	std::cout << "       \\XX\\        \\X/    \\XXX/    \\X/       /XX/    \n";
	std::cout << "           \\        |      \\X/      |       /     \n";
	std::cout << "\n\n";
	std::cout << "/////////////////////////////////////////////////////////////\n";
	std::cout << "/////////////////////////////////////////////////////////////\n";
	std::cout << "\n\n";
	return;
	}

/* print_rse
* Purpose: prints run, subrun event
* Parameters: run, subrun, event
* Returns: none
*/
void print_rse(int run, int subrun, int event){
    std::cout << "Run, Subrun, Event: " << run << " " << subrun << " " << event << "\n";
}

/* get_truth_info
* Purpose: gets the truth information from larlite
* Parameters: larlite storage manager, entry index, event id, plane pixel count vector
* Returns: a modified event/subrun that contains the number of protons, neutrons,
* charged and uncharged pions, CC/NC, flavor, interaction type, number of empty planes.
* See bottom of function for examples
*/
std::vector<int> get_truth_info(larlite::storage_manager* io_larlite, int entry, int event_wire, int subrun_wire, std::vector<int> pxl_count){
	//Now lets get all the truth information and print to std::cout
	io_larlite->go_to(entry);
	
	//load in input tree
	larlite::event_mctruth* ev_mctruth = (larlite::event_mctruth*)io_larlite->get_data(larlite::data::kMCTruth,  "generator" );

	// get event info
	// Neutrino energy
   float _enu_true=0.0;
   _enu_true = ev_mctruth->at(0).GetNeutrino().Nu().Momentum().E()*1000.0;
   std::cout<<"Neutrino Energy: "<<_enu_true<<std::endl;
   
   int flavors;
   int nc_cc;
   // cc or nc event?
   bool ccevent = false;
   int ccnc = ev_mctruth->at(0).GetNeutrino().CCNC();
   if (ccnc ==0 ) ccevent=true;
   if (ccevent){
	   std::cout<<"Is a CC Event"<<std::endl;
	   nc_cc = 1;
   } else {
	   std::cout<<"Is a NC Event"<<std::endl;
	   nc_cc = 0;
   }
   
   
   // type of Neutrino
   int nu_pdg = ev_mctruth->at(0).GetNeutrino().Nu().PdgCode();
   if (nu_pdg== 12){
	   std::cout<<"Muon Neutrino event "<<std::endl;
	   flavors = 0;
   } else if (nu_pdg== -12){
	   std::cout<<"Muon Anti Neutrino event "<<std::endl;
	   flavors = 1;
   } else if (nu_pdg== 14){
	   std::cout<<"Electron Neutrino event "<<std::endl;
	   flavors = 2;
   } 
   else if (nu_pdg== -14){
	   std::cout<<"Electon Anti Neutrino event "<<std::endl;
	   flavors = 3;
   }
   
   int interactionType;
   // type of interction - see comments at end of script
   int int_type= ev_mctruth->at(0).GetNeutrino().InteractionType();
   if (int_type == 1001 || int_type == 1002){
	   std::cout<<"QE Interaction "<<std::endl;
	   interactionType = 0;
   } else if (int_type >= 1003 && int_type <= 1090){
	   std::cout<<"RES Interaction "<<std::endl;
	   interactionType = 1;
   } else if (int_type == 1092 || int_type == 1091){
	   std::cout<<"DIS Interaction "<<std::endl;
	   interactionType = 2;
   } else{
	   std::cout<<"Other Interaction: "<<int_type<<std::endl;
	   interactionType = 3;
   }

   
   int num_protons = 0;
   int num_neutrons = 0;
   int num_pion_charged = 0;
   int num_pion_neutral = 0;
   for(int part =0;part<(int)ev_mctruth->at(0).GetParticles().size();part++){
	 // pick only final state particles
	 if (ev_mctruth->at(0).GetParticles().at(part).StatusCode() == 1){
	   int pdg = ev_mctruth->at(0).GetParticles().at(part).PdgCode();
	   if (pdg == 2212){
			float momentum = ev_mctruth->at(0).GetParticles().at(part).Momentum()[0];
			if (momentum >= 0.05) num_protons++;
	   }
	   else if (pdg == 2112){
		   float momentum = ev_mctruth->at(0).GetParticles().at(part).Momentum()[0];
		   if (momentum >= 0.05) num_neutrons++;
	   }
	   else if (pdg == 111 ) num_pion_neutral++;
	   else if (pdg == 211 || pdg == -211) num_pion_charged++;
   
	 }//end of if status = 1 statement
   }//end of loop over particles
   
   std::cout<<"Number of protons: "<<num_protons<<std::endl;
   std::cout<<"Number of neutrons: "<<num_neutrons<<std::endl;
   std::cout<<"Number of charged pions: "<<num_pion_charged<<std::endl;
   std::cout<<"Number of neutral pions: "<<num_pion_neutral<<std::endl;
   
   // makes event also store number of protons, neutrons, charged/uncharged pions
   // (in that order)
   // example: event = 66104301
   // event: 6610
   // protons: 4
   // neutrons: 3
   // charged pions: 0
   // uncharged pions: 1
   int event = event_wire;
   event = event*10000;
   event = event + num_protons*1000;
   event = event + num_neutrons*100;
   event = event + num_pion_charged*10;
   event = event + num_pion_neutral;
   
   // makes subrun also store NC/CC, the flavor of the ineraction, the interaction type, and how many planes are empty
   // NC/CC:
   // NC = 0
   // CC = 1
   //
   // Flavors:
   // muon neutrino= 0
   // muon antineutrino = 1
   // electron neutrino = 2
   // electron antineutrino = 3
   // 
   // interaction type:
   // QE = 0
   // RES = 1
   // DIS = 2
   // other = 3
   //
   // planes:
   // no planes empty = 0
   // 1 plane empty = 1
   // 2 plane empty = 2
   // 3 plane empty = 3
   //
   // example: subrun = 1290300
   // subrun: 129
   // NC/CC: NC
   // flavor: electron neutrino
   // interaction type: QE
   // planes: none empty
   int subrun = subrun_wire;
   subrun = subrun*10000;
   subrun = subrun + nc_cc*1000;
   subrun = subrun + flavors*100;
   subrun = subrun + interactionType*10;
   int num_empty_planes = 0;
   int good_entry = 1;
   for (int i = 0; i < 3; i++){
	   std::cout << "pxl_count for plane " << i << ": " << pxl_count[i] << "\n";
	   if (pxl_count[i] <= 60) good_entry = 0;
	   if (pxl_count[i] == 0) num_empty_planes++;
   }
   std::cout << "num_empty_planes: " << num_empty_planes << "\n";
   subrun = subrun + num_empty_planes;
   
   // TODO: change the subrun truth saving on dataLoader
   std::vector<int> truth = {event, subrun, good_entry};
   
   return truth;
}
