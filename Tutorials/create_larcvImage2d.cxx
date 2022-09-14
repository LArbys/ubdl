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
#include "larlite/LArUtil/LArProperties.h"
#include "larlite/LArUtil/DetectorProperties.h"
#include "larlite/LArUtil/Geometry.h"
#include "larlite/LArUtil/ClockConstants.h"
#include "larlite/LArUtil/SpaceChargeMicroBooNE.h"
// larcv
#include "larcv/core/DataFormat/IOManager.h"
#include "larcv/core/DataFormat/EventImage2D.h"
#include "larcv/core/DataFormat/EventPGraph.h"

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


	int start_entry = 0;
	int nentries_mc_cv = in_larcv->get_n_entries();
	std::cout << "Entries in File:  " << nentries_mc_cv << "\n";

  // loop to set entry if we only want one
  if (specific_entry != -1){
		start_entry = specific_entry;
		nentries_mc_cv = start_entry+1;
	}

  // loop through all the entries in the file
	for (int entry=start_entry; entry < nentries_mc_cv; entry++){

    std::cout << "Entry : " << entry << "\n\n";
		in_larcv->read_entry(entry);
		std::cout << "\n";

    // load in the event display image (contains the 3 planes)
    larcv::EventImage2D* ev_in_adc_dlreco  = (larcv::EventImage2D*)(in_larcv->get_data(larcv::kProductImage2D,"wire"));
    // take the larcv event image and change to a vector of images (one for each plane)
    std::vector< larcv::Image2D > img_2d_in_v = ev_in_adc_dlreco->Image2DArray();
    // we also get the meta. We will need a copy of this to save
    const std::vector<larcv::ImageMeta> meta_orig_v = {img_2d_in_v.at(0).meta(), img_2d_in_v.at(1).meta(), img_2d_in_v.at(2).meta() };

    // get identifying info of entry
    int run = ev_in_adc_dlreco->run();
		int subrun = ev_in_adc_dlreco->subrun();
		int event = ev_in_adc_dlreco->event();
		std::cout << run << " " << subrun << " " << event << "\n";

    // now create the output image2d
    larcv::EventImage2D* ev_out_adc_dlreco = (larcv::EventImage2D*)out_larcv->get_data(larcv::kProductImage2D,"example");
    ev_out_adc_dlreco->clear();
    std::vector<larcv::Image2D> out_v;

    // loop through all the pixels of the input to get the pixel information (number is standard uboone image size)
    for (int plane = 0; plane<3;plane++){
      // column = time tic, row = wire number
      // create single image2d for the plane
      larcv::Image2D single_out( meta_orig_v[plane] );
      single_out.paint(0.0);
      for (int col = 0; col<3456;col++){
        for (int row = 0; row<1008;row++){
          // get the pixel value for each plane
          double val = img_2d_in_v[plane].pixel(row,col);
          // now save to new image2d
          single_out.set_pixel(row,col,val);
        }//end of loop over rows
      }//end of loop over cols
      out_v.emplace_back( std::move(single_out) );
    }//end of loop over planes

    ev_out_adc_dlreco->Emplace( std::move(out_v) );
    out_larcv->set_id( run, subrun, event );
    out_larcv->save_entry();
	} //End of entry loop
  // close larcv I) manager
	in_larcv->finalize();
	delete in_larcv;
  out_larcv->reverse_all_products();
  out_larcv->finalize();
  delete out_larcv;

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
