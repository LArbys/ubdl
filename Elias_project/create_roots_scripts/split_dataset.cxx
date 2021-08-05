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
// std::vector<int> getProjectedPixel(const std::vector<double>& pos3d,
// 				   const larcv::ImageMeta& meta,
// 				   const int nplanes,
// 				   const float fracpixborder=1.5 );


// main fuction to load in a dlana root file, make event display image, get some truth.
int main(int nargs, char** argv){
    std::cout << "Hello world " << "\n";
	std::cout << "Args Required, Order Matters:\n";
	std::cout << "Inputfile with SparseImage\n";
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
  std::string output_file = output_dir + "SparseClassifierTrainingSet_3.root";
  int num_per_catagory = 30000;


    //create larcv storage Manager
	larcv::IOManager* io_larcv  = new larcv::IOManager(larcv::IOManager::kREAD,"IOManager_In", larcv::IOManager::kTickForward);
    larcv::IOManager* out_larcv  = new larcv::IOManager(larcv::IOManager::kWRITE,"IOManager_Out", larcv::IOManager::kTickForward);

    // load in and intialize larcv products
    // reversenecessary due to how products were created
    io_larcv->reverse_all_products();
  	io_larcv->add_in_file(input_file);
  	io_larcv->initialize();
    out_larcv->set_out_file(output_file);
    out_larcv->initialize();

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
		bool is_flav_CCNuMu = false;
		bool is_flav_CCNuE = false;
		bool is_flav_NCNuMu = false;
		bool is_flav_NCNuE = false;
		
		bool is_type_QE = false;
		bool is_type_RES = false;
		bool is_type_DIS = false;
		bool is_type_other = false;
		
		bool is_proton_0 = false;
		bool is_proton_1 = false;
		bool is_proton_2 = false;
		bool is_proton_g2 = false;
		
		bool is_neutron_0 = false;
		bool is_neutron_1 = false;
		bool is_neutron_2 = false;
		bool is_neutron_g2 = false;
		
		bool is_c_pion_0 = false;
		bool is_c_pion_1 = false;
		bool is_c_pion_2 = false;
		bool is_c_pion_g2 = false;
		
		bool is_n_pion_0 = false;
		bool is_n_pion_1 = false;
		bool is_n_pion_2 = false;
		bool is_n_pion_g2 = false;


    std::cout << "Entry : " << entry << "\n\n";
		io_larcv->read_entry(entry);
    
    larcv::EventSparseImage* ev_in_adc_dlreco  = (larcv::EventSparseImage*)(io_larcv->get_data(larcv::kProductSparseImage,"nbkrnd_sparse"));
	
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
            is_flav_CCNuE = true;
        } else if (flavors == 2 or flavors == 3){
			std::cout << "CC NuMu\n";
            is_flav_CCNuMu = true;
        }
    } else{// nc_cc == 0
        if (flavors == 0 or flavors == 1){
			std::cout << "NC NuE\n";
            is_flav_NCNuE = true;
        } else if (flavors == 2 or flavors == 3){
			std::cout << "NC NuMu\n";
            is_flav_NCNuMu = true;
        }
    }
    
    
    int interactionType = (_subrun%100)/10;
    std::cout << "interactionType: " << interactionType << "\n";
    if (interactionType == 0){
        std::cout << "QE Interaction\n";
        is_type_QE = true;
    } else if (interactionType == 1){
        std::cout << "RES Interaction\n";
        is_type_RES = true;
    } else if (interactionType == 2){
        std::cout << "DIS Interaction\n";
        is_type_DIS = true;
    } else if (interactionType == 3){
        std::cout << "Other Interaction\n";
        is_type_other = true;
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
        is_proton_0 = true;
    } else if (num_protons == 1){
        std::cout << "1 Protons\n";
        is_proton_1 = true;
    } else if (num_protons == 2){
        std::cout << "2 Protons\n";
        is_proton_2 = true;
    } else if (num_protons == 3){
        std::cout << ">2 Protons\n";
        is_proton_g2 = true;
    }
    
    int num_neutrons = (_event%1000)/100;
    if (num_neutrons > 2) num_neutrons = 3;
    std::cout << "num_neutrons: " << num_neutrons << "\n";
    if (num_neutrons == 0){
        std::cout << "0 Neutrons\n";
        is_neutron_0 = true;
    } else if (num_neutrons == 1){
        std::cout << "1 Neutrons\n";
        is_neutron_1 = true;
    } else if (num_neutrons == 2){
        std::cout << "2 Neutrons\n";
        is_neutron_2 = true;
    } else if (num_neutrons == 3){
        std::cout << ">2 Neutrons\n";
        is_neutron_g2 = true;
    }
    
    int num_pion_charged = (_event%100)/10;
    if (num_pion_charged > 2) num_pion_charged = 3;
    std::cout << "num_pion_charged: " << num_pion_charged << "\n";
    if (num_pion_charged == 0){
        std::cout << "0 Charged Pions\n";
        is_c_pion_0 = true;
    } else if (num_pion_charged == 1){
        std::cout << "1 Charged Pions\n";
        is_c_pion_1 = true;
    } else if (num_pion_charged == 2){
        std::cout << "2 Charged Pions\n";
        is_c_pion_2 = true;
    } else if (num_pion_charged == 3){
        std::cout << ">2 Charged Pions\n";
        is_c_pion_g2 = true;
    }
    
    int num_pion_neutral = _event%10;
    if (num_pion_neutral > 2) num_pion_neutral = 3;
    std::cout << "num_pion_neutral: " << num_pion_neutral << "\n";
    if (num_pion_neutral == 0){
        std::cout << "0 Neutral Pions\n";
        is_n_pion_0 = true;
    } else if (num_pion_neutral == 1){
        std::cout << "1 Neutral Pions\n";
        is_n_pion_1 = true;
    } else if (num_pion_neutral == 2){
        std::cout << "2 Neutral Pions\n";
        is_n_pion_2 = true;
    } else if (num_pion_neutral == 3){
        std::cout << ">2 Neutral Pions\n";
        is_n_pion_g2 = true;
    }
    
    int event = _event/10000;
    std::cout << "event: " << event << "\n";
    
	// decide if we need to save the entry:
	bool keep_entry = false;
	
	if (is_flav_CCNuMu and tot_flav_CCNuMu <= 100) keep_entry = true;
	else if (is_flav_CCNuE and tot_flav_CCNuE <= num_per_catagory) keep_entry = true;
	else if (is_flav_NCNuMu and tot_flav_NCNuMu <= num_per_catagory) keep_entry = true;
	else if (is_flav_NCNuE and tot_flav_NCNuE <= num_per_catagory) keep_entry = true;

	else if (is_type_QE and tot_type_QE <= num_per_catagory) keep_entry = true;
	else if (is_type_RES and tot_type_RES <= num_per_catagory) keep_entry = true;
	else if (is_type_DIS and tot_type_DIS <= num_per_catagory) keep_entry = true;
	else if (is_type_other and tot_type_other <= num_per_catagory) keep_entry = true;
	
	else if (is_proton_0 and tot_proton_0 <= num_per_catagory) keep_entry = true;
	else if (is_proton_1 and tot_proton_1 <= num_per_catagory) keep_entry = true;
	else if (is_proton_2 and tot_proton_2 <= num_per_catagory) keep_entry = true;
	else if (is_proton_g2 and tot_proton_g2 <= num_per_catagory) keep_entry = true;

	else if (is_neutron_0 and tot_neutron_0 <= num_per_catagory) keep_entry = true;
	else if (is_neutron_1 and tot_neutron_1 <= num_per_catagory) keep_entry = true;
	else if (is_neutron_2 and tot_neutron_2 <= num_per_catagory) keep_entry = true;
	else if (is_neutron_g2 and tot_neutron_g2 <= num_per_catagory) keep_entry = true;

	else if (is_c_pion_0 and tot_c_pion_0 <= 100) keep_entry = true;
	else if (is_c_pion_1 and tot_c_pion_1 <= num_per_catagory) keep_entry = true;
	else if (is_c_pion_2 and tot_c_pion_2 <= num_per_catagory) keep_entry = true;
	else if (is_c_pion_g2 and tot_c_pion_g2 <= num_per_catagory) keep_entry = true;

	else if (is_n_pion_0 and tot_n_pion_0 <= 100) keep_entry = true;
	else if (is_n_pion_1 and tot_n_pion_1 <= num_per_catagory) keep_entry = true;
	else if (is_n_pion_2 and tot_n_pion_2 <= num_per_catagory) keep_entry = true;
	else if (is_n_pion_g2 and tot_n_pion_g2 <= num_per_catagory) keep_entry = true;

	if (keep_entry){
        if (is_flav_CCNuMu) tot_flav_CCNuMu++;
        else if (is_flav_CCNuE) tot_flav_CCNuE++;
        else if (is_flav_NCNuMu) tot_flav_NCNuMu++;
        else if (is_flav_NCNuE) tot_flav_NCNuE++;

        if (is_type_QE) tot_type_QE++;
        else if (is_type_RES) tot_type_RES++;
        else if (is_type_DIS) tot_type_DIS++;
        else if (is_type_other) tot_type_other++;
        
        if (is_proton_0) tot_proton_0++;
        else if (is_proton_1) tot_proton_1++;
        else if (is_proton_2) tot_proton_2++;
        else if (is_proton_g2) tot_proton_g2++;

        if (is_neutron_0) tot_neutron_0++;
        else if (is_neutron_1) tot_neutron_1++;
        else if (is_neutron_2) tot_neutron_2++;
        else if (is_neutron_g2) tot_neutron_g2++;

        if (is_c_pion_0) tot_c_pion_0++;
        else if (is_c_pion_1) tot_c_pion_1++;
        else if (is_c_pion_2) tot_c_pion_2++;
        else if (is_c_pion_g2) tot_c_pion_g2++;

        if (is_n_pion_0) tot_n_pion_0++;
        else if (is_n_pion_1) tot_n_pion_1++;
        else if (is_n_pion_2) tot_n_pion_2++;
        else if (is_n_pion_g2) tot_n_pion_g2++;
        
        std::cout << "KEEPER\n";
        std::vector<larcv::SparseImage> out_sparse = ev_in_adc_dlreco->SparseImageArray();
        larcv::EventSparseImage* ev_out_adc_dlreco_sparse = (larcv::EventSparseImage*) out_larcv->get_data(larcv::kProductSparseImage,"nbkrnd_sparse");
        ev_out_adc_dlreco_sparse->Emplace( std::move(out_sparse) );
		out_larcv->set_id( _run, _subrun, _event);
        
        
        
        out_larcv->save_entry();

	}

	} //End of entry loop
  // // close larcv manager
	// io_larlite->close();
	// delete io_larlite;
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
    io_larcv->finalize();
	delete io_larcv;
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
