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
std::vector<int> getProjectedPixel(const std::vector<double>& pos3d,
				   const larcv::ImageMeta& meta,
				   const int nplanes,
				   const float fracpixborder=1.5 );


// main fuction to load in a dlana root file, make event display image, get some truth.
int main(int nargs, char** argv){
	std::cout << "Hello world " << "\n";
	std::cout << "Args Required, Order Matters:\n";
	std::cout << "Inputfile with Truth tree\n";
	std::cout << "Args Optional:\n";
	std::cout << "0 for truth info, 1 for entry length (default both)\n";


  if (nargs < 2){
		std::cout << "Not Enough Args\n";
		return 1;
	}
  std::string input_file = argv[1];

  int mode = -1;
    if (nargs > 2){
  	  mode = atoi(argv[2]);
    }
  //create larcv storage Manager
	larcv::IOManager* io_larcv  = new larcv::IOManager(larcv::IOManager::kREAD,"IOManager_Tagger", larcv::IOManager::kTickForward);
    // load in and intialize larcv products
    // reversenecessary due to how products were created
	io_larcv->reverse_all_products();
  	io_larcv->add_in_file(input_file);
  	io_larcv->initialize();
  	int start_entry = 0;
  	int nentries_mc_cv = io_larcv->get_n_entries();
    
    int tot_flav_CCNuMu = 0;
    int tot_flav_CCNuE = 0;
    int tot_flav_NCNuMu = 0;
    int tot_flav_NCNuE = 0;
    
    int tot_type_QE = 0;
    int tot_type_RES = 0;
    int tot_type_DIS = 0;
    int tot_type_other = 0;
    
    int tot_proton_0 = 0;
    int tot_proton_1 = 0;
    int tot_proton_2 = 0;
    int tot_proton_g2 = 0;
    
    int tot_neutron_0 = 0;
    int tot_neutron_1 = 0;
    int tot_neutron_2 = 0;
    int tot_neutron_g2 = 0;
    
    int tot_c_pion_0 = 0;
    int tot_c_pion_1 = 0;
    int tot_c_pion_2 = 0;
    int tot_c_pion_g2 = 0;
    
    int tot_n_pion_0 = 0;
    int tot_n_pion_1 = 0;
    int tot_n_pion_2 = 0;
    int tot_n_pion_g2 = 0;
    
    int tot_dead_planes = 0;
    
    

	std::cout << "Entries in File:  " << nentries_mc_cv << "\n";

  // loop through all the entries in the file
	for (int entry=start_entry; entry < nentries_mc_cv; entry++){

	if (mode != 1) std::cout << "Entry : " << entry << "\n\n";
		io_larcv->read_entry(entry);
    
    larcv::EventSparseImage* ev_in_adc_dlreco  = (larcv::EventSparseImage*)(io_larcv->get_data(larcv::kProductSparseImage,"nbkrnd_sparse"));
	
	if (mode == 0 or mode == -1){
		//print out r,s,e
	    int _run = ev_in_adc_dlreco->run();
		int _subrun = ev_in_adc_dlreco->subrun();
		int _event = ev_in_adc_dlreco->event();
	    std::cout<<"(r,s,e)="<<_run<<","<<_subrun<<","<<_event<<std::endl;
	    
	    std::cout << "raw subrun: " << _subrun << "\n";
	    int nc_cc = (_subrun%10000)/1000;
	    std::cout << "nc_cc: " << nc_cc << "\n";
	    int flavors = (_subrun%1000)/100;
	    std::cout << "flavors: " << flavors << "\n";
	    if (nc_cc == 1){
	        if (flavors == 0 or flavors == 1){
				std::cout << "CC NuE\n";
	            tot_flav_CCNuE++;
	        } else if (flavors == 2 or flavors == 3){
				std::cout << "CC NuMu\n";
	            tot_flav_CCNuMu++;
	        }
	    } else{// nc_cc == 0
	        if (flavors == 0 or flavors == 1){
				std::cout << "NC NuE\n";
	            tot_flav_NCNuE++;
	        } else if (flavors == 2 or flavors == 3){
				std::cout << "NC NuMu\n";
	            tot_flav_NCNuMu++;
	        }
	    }
	    
	    
	    int interactionType = (_subrun%100)/10;
	    std::cout << "interactionType: " << interactionType << "\n";
	    if (interactionType == 0){
	        std::cout << "QE Interaction\n";
	        tot_type_QE++;
	    } else if (interactionType == 1){
	        std::cout << "RES Interaction\n";
	        tot_type_RES++;
	    } else if (interactionType == 2){
	        std::cout << "DIS Interaction\n";
	        tot_type_DIS++;
	    } else if (interactionType == 3){
	        std::cout << "Other Interaction\n";
	        tot_type_other++;
	    }
	    
	    int planes = _subrun%10;
	    std::cout << "planes: " << planes << "\n";
	    if (planes != 0) tot_dead_planes++;
	    
	    int subrun = _subrun/10000;
	    std::cout << "subrun: " << subrun << "\n";
	    
	    
	    std::cout << "raw event: " << _event << "\n";
	    
	    int num_protons = (_event%10000)/1000;
	    if (num_protons > 2) num_protons = 3;
	    std::cout << "num_protons: " << num_protons << "\n";
	    if (num_protons == 0){
	        std::cout << "0 Protons\n";
	        tot_proton_0++;
	    } else if (num_protons == 1){
	        std::cout << "1 Protons\n";
	        tot_proton_1++;
	    } else if (num_protons == 2){
	        std::cout << "2 Protons\n";
	        tot_proton_2++;
	    } else if (num_protons == 3){
	        std::cout << ">2 Protons\n";
	        tot_proton_g2++;
	    }
	    
	    int num_neutrons = (_event%1000)/100;
	    if (num_neutrons > 2) num_neutrons = 3;
	    std::cout << "num_neutrons: " << num_neutrons << "\n";
	    if (num_neutrons == 0){
	        std::cout << "0 Neutrons\n";
	        tot_neutron_0++;
	    } else if (num_neutrons == 1){
	        std::cout << "1 Neutrons\n";
	        tot_neutron_1++;
	    } else if (num_neutrons == 2){
	        std::cout << "2 Neutrons\n";
	        tot_neutron_2++;
	    } else if (num_neutrons == 3){
	        std::cout << ">2 Neutrons\n";
	        tot_neutron_g2++;
	    }
	    
	    int num_pion_charged = (_event%100)/10;
	    if (num_pion_charged > 2) num_pion_charged = 3;
	    std::cout << "num_pion_charged: " << num_pion_charged << "\n";
	    if (num_pion_charged == 0){
	        std::cout << "0 Charged Pions\n";
	        tot_c_pion_0++;
	    } else if (num_pion_charged == 1){
	        std::cout << "1 Charged Pions\n";
	        tot_c_pion_1++;
	    } else if (num_pion_charged == 2){
	        std::cout << "2 Charged Pions\n";
	        tot_c_pion_2++;
	    } else if (num_pion_charged == 3){
	        std::cout << ">2 Charged Pions\n";
	        tot_c_pion_g2++;
	    }
	    
	    int num_pion_neutral = _event%10;
	    if (num_pion_neutral > 2) num_pion_neutral = 3;
	    std::cout << "num_pion_neutral: " << num_pion_neutral << "\n";
	    if (num_pion_neutral == 0){
	        std::cout << "0 Neutral Pions\n";
	        tot_n_pion_0++;
	    } else if (num_pion_neutral == 1){
	        std::cout << "1 Neutral Pions\n";
	        tot_n_pion_1++;
	    } else if (num_pion_neutral == 2){
	        std::cout << "2 Neutral Pions\n";
	        tot_n_pion_2++;
	    } else if (num_pion_neutral == 3){
	        std::cout << ">2 Neutral Pions\n";
	        tot_n_pion_g2++;
	    }
	    
	    int event = _event/10000;
	    std::cout << "event: " << event << "\n";
	}
	
	if (mode == 1 or mode == -1){
		std::cout << ev_in_adc_dlreco->SparseImageArray()[0].len() << "\n";
	}
    
	} //End of entry loop
  // // close larcv manager
	// io_larlite->close();
	// delete io_larlite;
    if (mode == 0 or mode == -1){
		std::cout << "\n\n";
	    std::cout << "TOTALS:\n\n";
	    std::cout << "Total Entries: " << nentries_mc_cv << "\n";
	    std::cout << "\n";
	    std::cout << "Total CC NuMu: " << tot_flav_CCNuMu << "\n";
	    std::cout << "Total CC NuE: " << tot_flav_CCNuE << "\n";
	    std::cout << "Total NC NuMu: " << tot_flav_NCNuMu << "\n";
	    std::cout << "Total NC NuE: " << tot_flav_NCNuE << "\n";
	    std::cout << "\n";
	    std::cout << "Total QE Interaction: " << tot_type_QE << "\n";
	    std::cout << "Total RES Interaction: " << tot_type_RES << "\n";
	    std::cout << "Total DIS Interaction: " << tot_type_DIS << "\n";
	    std::cout << "Total Other Interaction: " << tot_type_other << "\n";
	    std::cout << "\n";
	    std::cout << "Total 0 Proton Events: " << tot_proton_0 << "\n";
	    std::cout << "Total 1 Proton Events: " << tot_proton_1 << "\n";
	    std::cout << "Total 2 Proton Events: " << tot_proton_2 << "\n";
	    std::cout << "Total >2 Proton Events: " << tot_proton_g2 << "\n";
	    std::cout << "\n";
	    std::cout << "Total 0 Neutron Events: " << tot_neutron_0 << "\n";
	    std::cout << "Total 1 Neutron Events: " << tot_neutron_1 << "\n";
	    std::cout << "Total 2 Neutron Events: " << tot_neutron_2 << "\n";
	    std::cout << "Total >2 Neutron Events: " << tot_neutron_g2 << "\n";
	    std::cout << "\n";
	    std::cout << "Total 0 Charged Pion Events: " << tot_c_pion_0 << "\n";
	    std::cout << "Total 1 Charged Pion Events: " << tot_c_pion_1 << "\n";
	    std::cout << "Total 2 Charged Pion Events: " << tot_c_pion_2 << "\n";
	    std::cout << "Total >2 Charged Pion Events: " << tot_c_pion_g2 << "\n";
	    std::cout << "\n";
	    std::cout << "Total 0 Neutral Pion Events: " << tot_n_pion_0 << "\n";
	    std::cout << "Total 1 Neutral Pion Events: " << tot_n_pion_1 << "\n";
	    std::cout << "Total 2 Neutral Pion Events: " << tot_n_pion_2 << "\n";
	    std::cout << "Total >2 Neutral Pion Events: " << tot_n_pion_g2 << "\n";
	    std::cout << "\n";
	    std::cout << "Total Dead Planes: " << tot_dead_planes << "\n";
	    std::cout << "\n";
	}
	
	
    io_larcv->finalize();
	delete io_larcv;
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

//   kUnknown                   =   0
// kCCQE                      =   1
// kNCQE                      =   2
// kResCCNuProtonPiPlus       =   3
// kResCCNuNeutronPi0         =   4
// kResCCNuNeutronPiPlus      =   5
// kResNCNuProtonPi0          =   6
// kResNCNuProtonPiPlus       =   7
// kResNCNuNeutronPi0         =   8
// kResNCNuNeutronPiMinus     =   9
// kResCCNuBarNeutronPiMinus  =   10
// kResCCNuBarProtonPi0       =   11
// kResCCNuBarProtonPiMinus   =   12
// kResNCNuBarProtonPi0       =   13
// kResNCNuBarProtonPiPlus    =   14
// kResNCNuBarNeutronPi0      =   15
// kResNCNuBarNeutronPiMinus  =   16
// kResCCNuDeltaPlusPiPlus    =   17
// kResCCNuDelta2PlusPiMinus  =   21
// kResCCNuBarDelta0PiMinus   =   28
// kResCCNuBarDeltaMinusPiPlus=   32
// kResCCNuProtonRhoPlus      =   39
// kResCCNuNeutronRhoPlus     =   41
// kResCCNuBarNeutronRhoMinus =   46
// kResCCNuBarNeutronRho0     =   48
// kResCCNuSigmaPlusKaonPlus  =   53
// kResCCNuSigmaPlusKaon0     =   55
// kResCCNuBarSigmaMinusKaon0 =   60
// kResCCNuBarSigma0Kaon0     =   62
// kResCCNuProtonEta          =   67
// kResCCNuBarNeutronEta      =   70
// kResCCNuKaonPlusLambda0    =   73
// kResCCNuBarKaon0Lambda0    =   76
// kResCCNuProtonPiPlusPiMinus=   79
// kResCCNuProtonPi0Pi0       =   80
// kResCCNuBarNeutronPiPlusPiMinus =   85
// kResCCNuBarNeutronPi0Pi0   =   86
// kResCCNuBarProtonPi0Pi0    =   90
// kCCDIS                     =   91
// kNCDIS                     =   92
// kUnUsed1                   =   93
// kUnUsed2                   =   94
// kCCQEHyperon               =   95
// kNCCOH                     =   96
// kCCCOH                     =   97
// kNuElectronElastic         =   98
// kInverseMuDecay            =   99
