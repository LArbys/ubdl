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


  if (nargs < 2){
		std::cout << "Not Enough Args\n";
		return 1;
	}
  std::string input_file = argv[1];


  //createlarlite storage Manager
	larlite::storage_manager* io_larlite  = new larlite::storage_manager(larlite::storage_manager::kREAD);
  // load in and intialize larlite products
	io_larlite->add_in_filename(input_file);
	io_larlite->open();
	int start_entry = 0;
	int nentries_mc_cv = io_larlite->get_entries();

	std::cout << "Entries in File:  " << nentries_mc_cv << "\n";

  // loop through all the entries in the file
	for (int entry=start_entry; entry < nentries_mc_cv; entry++){

    std::cout << "Entry : " << entry << "\n\n";
		io_larlite->go_to(entry);

    //print out r,s,e
    int _run    = io_larlite->run_id();
    int _subrun = io_larlite->subrun_id();
    int _event  = io_larlite->event_id();
    std::cout<<"(r,s,e)="<<_run<<","<<_subrun<<","<<_event<<std::endl;

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
	} //End of entry loop
  // close larcv manager
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
