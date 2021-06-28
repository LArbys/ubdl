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

// initalize helper functions
void print_signal();
std::vector<int> getProjectedPixel(const std::vector<double>& pos3d,
				   const larcv::ImageMeta& meta,
				   const int nplanes,
				   const float fracpixborder=1.5 );


// main fuction to load in a dlana root file, make event display image, get some truth.
int main(int nargs, char** argv){
	std::cout << "Hello world " << "\n";
	std::cout << "Args Required, Order Matters:\n";
	std::cout << "Inputfile with Image2D\n";
	std::cout << "Output Directory\n";
	std::cout << "\n";
	std::cout << "Args Optional:\n";
	std::cout << "Specific Entry to Do (Default is all entries)\n";
	std::cout << "Custom String to Add to Output pngs\n";

  if (nargs < 3){
		std::cout << "Not Enough Args\n";
		return 1;
	}
  std::string input_file = argv[1];
	std::string output_dir = argv[2];
	std::string output_custom = "";
  int specific_entry = -1;
	if (nargs > 3){
		specific_entry = atoi(argv[3]);
	}
	if (nargs > 4){
		output_custom = argv[4];
	}

	std::cout << "Running with Args:\n";
	for (int i = 0; i<nargs;i++){
		std::cout << "Arg " << i << " : " << argv[i] << "\n";
	}

  // root style setting
	gStyle->SetOptStat(0);
  //create larcv IO IOManager
	larcv::IOManager* io_larcv  = new larcv::IOManager(larcv::IOManager::kREAD,"IOManager_Tagger", larcv::IOManager::kTickForward);
  // load in and intialize larcv products
  // reversenecessary due to how products were created
  //io_larcv->reverse_all_products();
	io_larcv->add_in_file(input_file);
	io_larcv->initialize();
	int start_entry = 0;
	int nentries_mc_cv = io_larcv->get_n_entries();
	std::cout << "Entries in File:  " << nentries_mc_cv << "\n";

  // loop to set entry if we only want one
  if (specific_entry != -1){
		start_entry = specific_entry;
		nentries_mc_cv = start_entry+1;
	}

  // loop through all the entries in the file
	for (int entry=start_entry; entry < nentries_mc_cv; entry++){
    // initialize root histograms
    // 2d histogram of doubles (name of object, title of plot, xbins, xmin, xmax, ybins, ymin, ymax)
		TH2D ev_disp_raw_u =TH2D("ev_disp_raw_u","ev_disp_raw_u ",3456,0,3456.,1008,0,1008.);
		TH2D ev_disp_raw_v =TH2D("ev_disp_raw_v","ev_disp_raw_v ",3456,0,3456.,1008,0,1008.);
		TH2D ev_disp_raw_y =TH2D("ev_disp_raw_y","ev_disp_raw_y ",3456,0,3456.,1008,0,1008.);
    // vertex only histograms
		TH2D vtx_u =TH2D("vtx_u","vtx_u ",3456,0,3456.,1008,0,1008.);
		TH2D vtx_v =TH2D("vtx_v","vtx_v ",3456,0,3456.,1008,0,1008.);
		TH2D vtx_y =TH2D("vtx_y","vtx_y ",3456,0,3456.,1008,0,1008.);
    std::cout << "Entry : " << entry << "\n\n";
		io_larcv->read_entry(entry);
		std::cout << "\n";

    // load in the event display image (contains the 3 planes)
    larcv::EventImage2D* ev_in_adc_dlreco  = (larcv::EventImage2D*)(io_larcv->get_data(larcv::kProductImage2D,"wire"));

    // get identifying info of entry
    int run = ev_in_adc_dlreco->run();
		int subrun = ev_in_adc_dlreco->subrun();
		int event = ev_in_adc_dlreco->event();
		std::cout << run << " " << subrun << " " << event << "\n";

    // take the larcv event image and change to a vector of images (one for each plane)
    std::vector< larcv::Image2D > img_2d_in_v = ev_in_adc_dlreco->Image2DArray();
    // loop through all the pixels to get the pixel information (number is standard uboone image size)
    // column = time tic, row = wire number
	  for (int col = 0; col<3456;col++){
	    for (int row = 0; row<1008;row++){
        // get the pixel value for each plane
	      double val_u = img_2d_in_v[0].pixel(row,col);
	      double val_v = img_2d_in_v[1].pixel(row,col);
	      double val_y = img_2d_in_v[2].pixel(row,col);

        // reset very high values - don't do if using in algorithm, but useful for display
				if (val_u > 100.) val_u = 100.0;
				if (val_v > 100.) val_v = 100.0;
				if (val_y > 100.) val_y = 100.0;
        // set 0 values to slightly non-zero for better display
        if (val_u == 0) val_u = 1.0;
				if (val_v == 0) val_v = 1.0;
				if (val_y == 0) val_y = 1.0;

        // fill root histograms
        ev_disp_raw_u.SetBinContent(col,row,val_u);
	      ev_disp_raw_v.SetBinContent(col,row,val_v);
	      ev_disp_raw_y.SetBinContent(col,row,val_y);
	    }
	  }

    // now fill reconsructed vertex histograms, will only fill a point.
    // first load event vertex object
    
	// larcv::EventPGraph* ev_test_pgraph  = (larcv::EventPGraph*)(io_larcv->get_data(larcv::kProductPGraph,"test"));
    // // turn larcv object into an object
    // std::vector<larcv::PGraph> test_pgraph_v = ev_test_pgraph->PGraphArray();
    // for (auto pgraph : test_pgraph_v) {
    //   for (larcv::ROI reco_roi : pgraph.ParticleArray()){
    //     // get 3d location of reco vertex
    //     std::vector<double> reco_vertex = {reco_roi.X(), reco_roi.Y(), reco_roi.Z()};
    //     // now project into 2d
    //     std::vector<int> vtx_rc = getProjectedPixel(reco_vertex, img_2d_in_v[0].meta(), 3);
    //     // fill root histograms
    //     vtx_u.Fill(vtx_rc[1], vtx_rc[0]);
    //     vtx_v.Fill(vtx_rc[2], vtx_rc[0]);
    //     vtx_y.Fill(vtx_rc[3], vtx_rc[0]);
    //     std::cout<<"FOUND A VERTEX"<<std::endl;
    //   }
    // }


    // display root histograms on a canvas
    // set some display options
    // save to a png (seperate image for each plane)
    TCanvas can("can", "histograms ", 3456, 1008);
	  can.cd();
    // first draw charge image
	  ev_disp_raw_u.SetTitle(Form("Wire Image U Plane Run: %d Subrun: %d Event: %d",run,subrun,event));
	  ev_disp_raw_u.SetXTitle("Column (Wire)");
	  ev_disp_raw_u.SetYTitle("Row (6 Ticks)");
	  ev_disp_raw_u.SetOption("COLZ");
	  ev_disp_raw_u.Draw("");
    // overlay vertex historgrams
    vtx_u.SetMarkerStyle(kStar);
    vtx_u.SetMarkerColor(2);
    vtx_u.SetMarkerSize(5);
    vtx_u.Draw("SAME");
		can.SaveAs(Form("%s/output_%d_%d_%d_u%s.png",output_dir.c_str(),run,subrun,event,output_custom.c_str()));

    ev_disp_raw_v.SetTitle(Form("Wire Image V Plane Run: %d Subrun: %d Event: %d",run,subrun,event));
	  ev_disp_raw_v.SetXTitle("Column (Wire)");
	  ev_disp_raw_v.SetYTitle("Row (6 Ticks)");
	  ev_disp_raw_v.SetOption("COLZ");
	  ev_disp_raw_v.Draw("");
    vtx_v.SetMarkerStyle(kStar);
    vtx_v.SetMarkerColor(2);
    vtx_u.SetMarkerSize(5);
    vtx_v.Draw("SAME");
		can.SaveAs(Form("%s/output_%d_%d_%d_v%s.png",output_dir.c_str(),run,subrun,event,output_custom.c_str()));

    ev_disp_raw_y.SetTitle(Form("Wire Image Y Plane Run: %d Subrun: %d Event: %d",run,subrun,event));
	  ev_disp_raw_y.SetXTitle("Column (Wire)");
	  ev_disp_raw_y.SetYTitle("Row (6 Ticks)");
	  ev_disp_raw_y.SetOption("COLZ");
	  ev_disp_raw_y.Draw("");
    vtx_y.SetMarkerStyle(kStar);
    vtx_y.SetMarkerColor(2);
    vtx_u.SetMarkerSize(5);
    vtx_y.Draw("SAME");
    can.SaveAs(Form("%s/output_%d_%d_%d_y%s.png",output_dir.c_str(),run,subrun,event,output_custom.c_str()));

	} //End of entry loop
  // close larcv I) manager
	io_larcv->finalize();
	delete io_larcv;
  print_signal();
  return 0;
	}//End of main

  std::vector<int> getProjectedPixel( const std::vector<double>& pos3d,
  				    const larcv::ImageMeta& meta,
  				    const int nplanes,
  				    const float fracpixborder ) {

    // function that takes in 3d positions and returns time,wire_u,wire_v,wire_y in pixel coordinates
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
  }//end of project pixel function

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
