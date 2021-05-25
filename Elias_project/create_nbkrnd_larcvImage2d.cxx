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
  std::string output_file = output_dir + "outputimage2d.root";
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
	
	
    // these are used to determine how many digits I should leave for the id
	// int max_protons = 0;
	// int max_neutrons = 0;
	// int max_pion_charged = 0;
	// int max_pion_neutral = 0;


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
		std::cout << "Wire: " << run_wire << " " << subrun_wire << " " << event_wire << "\n";
	int run_thrumu = ev_in_adc_dlreco_thrumu->run();
		int subrun_thrumu = ev_in_adc_dlreco_thrumu->subrun();
		int event_thrumu = ev_in_adc_dlreco_thrumu->event();
		std::cout << "thrumu: " << run_thrumu << " " << subrun_thrumu << " " << event_thrumu << "\n";



		//Now lets get all the truth information and print to std::cout
		io_larlite->go_to(entry);
		
		//load in input tree
	    larlite::event_mctruth* ev_mctruth = (larlite::event_mctruth*)io_larlite->get_data(larlite::data::kMCTruth,  "generator" );
	
	    // get event info
	    // Neutrino energy
	   float _enu_true=0.0;
	   _enu_true = ev_mctruth->at(0).GetNeutrino().Nu().Momentum().E()*1000.0;
	   std::cout<<"Neutrino Energy: "<<_enu_true<<std::endl;
	   
	   // cc or nc event?
	   bool ccevent = false;
	   int ccnc = ev_mctruth->at(0).GetNeutrino().CCNC();
	   if (ccnc ==0 ) ccevent=true;
	   if (ccevent) std::cout<<"Is a CC Event"<<std::endl;
	   else std::cout<<"Is a NC Event"<<std::endl;
	   
	   // type of Neutrino
	   int nu_pdg = ev_mctruth->at(0).GetNeutrino().Nu().PdgCode();
	   if (nu_pdg== 12) std::cout<<"Muon Neutrino event "<<std::endl;
	   else if (nu_pdg== -12) std::cout<<"Muon Anti Neutrino event "<<std::endl;
	   else if (nu_pdg== 14) std::cout<<"Electron Neutrino event "<<std::endl;
	   else if (nu_pdg== -14) std::cout<<"Electon Anti Neutrino event "<<std::endl;
	   
	   // type of interction - see comments at end of script
	   int int_type= ev_mctruth->at(0).GetNeutrino().InteractionType();
	   if (int_type == 1001 || int_type == 1002) std::cout<<"QE Interaction "<<std::endl;
	   else if (int_type >= 1003 && int_type <= 1090) std::cout<<"RES Interaction "<<std::endl;
	   else if (int_type == 1092 || int_type == 1091) std::cout<<"DIS Interaction "<<std::endl;
	   else std::cout<<"Other Interaction: "<<int_type<<std::endl;
	   
	   int num_protons = 0;
	   int num_neutrons = 0;
	   int num_pion_charged = 0;
	   int num_pion_neutral = 0;
	   for(int part =0;part<(int)ev_mctruth->at(0).GetParticles().size();part++){
	     // pick only final state particles
	     if (ev_mctruth->at(0).GetParticles().at(part).StatusCode() == 1){
	       int pdg = ev_mctruth->at(0).GetParticles().at(part).PdgCode();
	       if (pdg == 2212) num_protons++;
	       else if (pdg == 2112) num_neutrons++;
	       else if (pdg == 111 ) num_pion_neutral++;
	       else if (pdg == 211 || pdg == -211) num_pion_charged++;
	   
	     }//end of if status = 1 statement
	   }//end of loop over particles
	   
	   std::cout<<"Number of protons: "<<num_protons<<std::endl;
	   std::cout<<"Number of neutrons: "<<num_neutrons<<std::endl;
	   std::cout<<"Number of charged pions: "<<num_pion_charged<<std::endl;
	   std::cout<<"Number of neutral pions: "<<num_pion_neutral<<std::endl;
	   
	    std::cout << "\n";
		
		// used to find the largest number of protons, neutrons, charged/neutral
		// pions. Will use this information to determine how many digits to
		// allocate in the id
		// if (num_protons > max_protons){
		// 	max_protons = num_protons;
		// }
		// if (num_neutrons > max_neutrons){
		// 	max_neutrons = num_neutrons;
		// }
		// if (num_pion_charged > max_pion_charged){
		// 	max_pion_charged = num_pion_charged;
		// }
		// if (num_pion_neutral > max_pion_neutral){
		// 	max_pion_neutral = num_pion_neutral;
		// }


    // now create the output image2d
    larcv::EventImage2D* ev_out_adc_dlreco = (larcv::EventImage2D*)out_larcv->get_data(larcv::kProductImage2D,"mask_wire");
    ev_out_adc_dlreco->clear();
    std::vector<larcv::Image2D> out_v;
	unsigned int npixels = 0;// used to check total pixel count of image2d and sparse

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
		  
		  if (img_2d_in_v_thrumu[plane].pixel(row,col) != 0){
			  val = 0;
		  }
		  if (val >= 10){
			  npixels++;
		  }
		  
          // now save to new image2d
          single_out.set_pixel(row,col,val);
        }//end of loop over rows
      }//end of loop over cols
      out_v.emplace_back( std::move(single_out) );
    }//end of loop over planes

	// makes event also store number of protons, neutrons, charged/uncharged pions
	// (in that order)
	// example: event = 6610110301
	// event: 6610
	// protons: 11
	// neutrons: 03
	// charged pions: 0
	// uncharged pions: 1
	long int event = event_wire;
	event = event*1000000;
	event = event + num_protons*10000;
	event = event + num_neutrons*100;
	event = event + num_pion_charged*10;
	event = event + num_pion_neutral;
	std::cout << "event: " << event << "\n\n";
	
	std::vector<larcv::Image2D> out_v_s = out_v;
	
    ev_out_adc_dlreco->Emplace( std::move(out_v) );
    
	
	// Saving sparse:
	std::vector<float> thresholds;
	thresholds.push_back(10);
	thresholds.push_back(10);
	thresholds.push_back(10);
	// std::vector<larcv::Image2D*> out_v_p;
	// for (int plane = 0; plane<3;plane++){
	// 	out_v_p[plane] = &out_v[plane];
	// } 
	// larcv::EventImage2D* ev_out_adc_dlreco = (larcv::EventImage2D*)out_larcv->get_data(larcv::kProductImage2D,"mask_wire");

	std::vector<int> empty;
	larcv::EventSparseImage* ev_out_adc_dlreco_sparse = (larcv::EventSparseImage*) out_larcv->get_data(larcv::kProductSparseImage,"nbkrnd_wire");
	// ev_out_adc_dlreco_sparse->clear();
	larcv::SparseImage out_sparse(out_v_s, thresholds);
	
	// check to make sure image2D and SparseImage have the same number of pixels,
	// both with a threshold of 1
	int sPixels_count = 0;
	std::vector<float> sPixel_v = out_sparse.pixellist();
	for (int i = 0; i < sPixel_v.size(); i++){
		// std::cout << "sPixel_v: " << sPixel_v[i] << "\n";
		if (sPixel_v[i] >= 10){
			sPixels_count++;
		}
	}
	int sPixels_size = sPixel_v.size();
	
	
	// Now lets make an image2D from the sparse and count it's pixels:
	std::vector<larcv::Image2D> pixel2d = out_sparse.as_Image2D();
	unsigned int pcount = 0;
	for (int plane = 0; plane<3;plane++){
		for (int col = 0; col<3456;col++){
			for (int row = 0; row<1008;row++){
				double pval = pixel2d[plane].pixel(row,col);
				if (pval >= 10){
					pcount++;
				}
			}
		}
	}
	
	std::cout << "Pixel Count Check:\n";
	std::cout << "image2d pixel count: " << npixels << "\n";
	std::cout << "sparse pixel size: " << sPixels_size << "\n";
	std::cout << "sparse pixel count: " << sPixels_count << "\n";
	std::cout << "image2d from sparse pcount: " << pcount << "\n";
	if (pcount != npixels){
		std::cout << "PROBLEM: image2d pixel counts don't match\n";
	} else {
		std::cout << "success: image2d pixel counts match!\n";
	}
	if (sPixels_size != npixels || sPixels_count != npixels){
		std::cout << "PROBLEM: pixel counts are not the same\n";
		if (sPixels_size == 5*npixels){
			std::cout << ".size() gets 5*image2d\n";
		} else {
			std::cout << ".size() does not get 5*image2d\n";;
		}
		if (sPixels_count == 3*npixels){
			std::cout << "looping gets 3*image2d\n";
		} else {
			std::cout << "looping does not get 3*image2d\n";
		}
	}
	
	std::cout << "\n";

	
	
	
	ev_out_adc_dlreco_sparse->Emplace( std::move(out_sparse) );
	
	out_larcv->set_id( run_wire, subrun_wire, event);
    out_larcv->save_entry();
	
	
	
	} //End of entry loop
	
	// std::cout<<"Max number of protons: "<<max_protons<<std::endl;
	// std::cout<<"Max number of neutrons: "<<max_neutrons<<std::endl;
	// std::cout<<"Max number of charged pions: "<<max_pion_charged<<std::endl;
	// std::cout<<"Max number of neutral pions: "<<max_pion_neutral<<std::endl;
	// 
	//  std::cout << "\n";
	
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
