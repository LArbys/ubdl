#include <iostream>
#include <map>
#include <utility>
#include <vector>
// ROOT
#include "TFile.h"
#include "TTree.h"
#include "TH2F.h"
#include "TH1F.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TGraph.h"
#include "TLegend.h"
#include "TAttMarker.h"

TFile* fin;
TTree* tree;
int run=-1;
int subrun=-1;
int event=-1;

int num_numu;
int num_nue;
int num_good_vtx;
int num_bad_vtx;
int num_good_vtx_tagger;
int num_bad_vtx_tagger;
int num_good_vtx_lf;
int num_bad_vtx_lf;
std::vector<double>* _scex = NULL;
std::vector<int>* scenupixel = NULL; //tick u v y wire
std::vector<int>* nupixel = NULL; //tick u v y wire

int interaction_mode;
int ccnc;
int num_p;
int num_e;
int num_pi0;
int num_piplus;
int num_piminus;
int num_n;
std::vector<std::vector<int>>* good_vtx_pixel = NULL;
std::vector<std::vector<int>>* bad_vtx_pixel = NULL;

std::vector<std::vector<int>>* good_vtx_tag_pixel = NULL;
std::vector<std::vector<int>>* bad_vtx_tag_pixel = NULL;
std::vector<float>* p_enedep = NULL;
std::vector<float>* e_enedep = NULL;
float nu_energy;


void read_vtxana(){
    gStyle->SetOptStat(0);
    fin = new TFile("cosmictag_new_hadd.root","read");
    tree = (TTree*)fin->Get("tree");

    TH2F* huwo = NULL;
    TH2F* hvwo = NULL;
    TH2F* hywo = NULL;
    // TH2F* hywi = NULL;
    TH2F* hclusterid = NULL;
    TH2F* hcnufrac = NULL;
    TH2F* hcroi = NULL;
    TH2F* hssnetbg = NULL;
    TH2F* hssnetshower = NULL;
    TH2F* hssnettrack = NULL;
    TH2F* hearly = NULL;
    TH2F* hlate = NULL;
    TH2F* hdwall = NULL;

    tree->SetBranchAddress("hYwoTagger", &hywo);
    // tree->SetBranchAddress("hYwiTagger", &hywi);

    tree->SetBranchAddress("hUwoTagger", &huwo);
    tree->SetBranchAddress("hVwoTagger", &hvwo);

    tree->SetBranchAddress("hcroi", &hcroi);
    tree->SetBranchAddress("hssnetbg", &hssnetbg);
    tree->SetBranchAddress("hearly", &hearly);
    tree->SetBranchAddress("hlate", &hlate);
    tree->SetBranchAddress("hdwall", &hdwall);

    tree->SetBranchAddress("run",&run);
    tree->SetBranchAddress("subrun",&subrun);
    tree->SetBranchAddress("event",&event);

    tree->SetBranchAddress("num_numu",&num_numu);
    tree->SetBranchAddress("num_nue",&num_nue);
    tree->SetBranchAddress("num_e",&num_e);
    tree->SetBranchAddress("num_p",&num_p);
    tree->SetBranchAddress("num_pi0",&num_pi0);
    tree->SetBranchAddress("num_n",&num_n);
    tree->SetBranchAddress("num_piplus",&num_piplus);
    tree->SetBranchAddress("num_piminus",&num_piminus);

    tree->SetBranchAddress("p_enedep",&p_enedep);
    tree->SetBranchAddress("e_enedep",&e_enedep);

    tree->SetBranchAddress("num_good_vtx",&num_good_vtx);
    tree->SetBranchAddress("num_bad_vtx",&num_bad_vtx);
    tree->SetBranchAddress("num_good_vtx_tagger",&num_good_vtx_tagger);
    tree->SetBranchAddress("num_bad_vtx_tagger",&num_bad_vtx_tagger);

    tree->SetBranchAddress("_scex",&_scex);
    tree->SetBranchAddress("nupixel",&nupixel);
    tree->SetBranchAddress("scenupixel",&scenupixel);
    tree->SetBranchAddress("CCNC",&ccnc);
    tree->SetBranchAddress("interaction_mode",&interaction_mode);
    tree->SetBranchAddress("good_vtx_pixel",&good_vtx_pixel);
    tree->SetBranchAddress("bad_vtx_pixel",&bad_vtx_pixel);

    tree->SetBranchAddress("good_vtx_tag_pixel",&good_vtx_tag_pixel);
    tree->SetBranchAddress("bad_vtx_tag_pixel",&bad_vtx_tag_pixel);
    int tot_nu=0;
    int tot_has_vtx=0;
    int tot_has_vtx_tag=0;
    int tot_has_vtx_lf=0;
    int tot_has_gvtx=0;
    int tot_has_gvtx_tag=0;
    int tot_has_gvtx_lf=0;
    int tot_gvtx=0;
    int tot_bvtx=0;
    int tot_gvtx_tag=0;
    int tot_bvtx_tag=0;
    int tot_gvtx_lf=0;
    int tot_bvtx_lf=0;
    int tot_1e1p = 0;
    int tot_ccqe = 0;
    int tot_mec = 0;
    int tot_other = 0;
    int tot_ccqe_has_vtx = 0;
    int tot_ccqe_has_gvtx = 0;
    int tot_ccqe_bvtx = 0;
    int tot_mec_has_vtx = 0;
    int tot_mec_has_gvtx = 0;
    int tot_mec_bvtx = 0;
    int tot_other_has_vtx = 0;
    int tot_other_has_gvtx = 0;
    int tot_other_bvtx = 0;

    int tot_ccqe_has_vtx_tag = 0;
    int tot_ccqe_has_gvtx_tag = 0;
    int tot_ccqe_bvtx_tag = 0;
    int tot_mec_has_vtx_tag = 0;
    int tot_mec_has_gvtx_tag = 0;
    int tot_mec_bvtx_tag = 0;
    int tot_other_has_vtx_tag = 0;
    int tot_other_has_gvtx_tag = 0;
    int tot_other_bvtx_tag = 0;

    int tot_ccqe_has_vtx_lf = 0;
    int tot_ccqe_has_gvtx_lf = 0;
    int tot_ccqe_bvtx_lf = 0;
    int tot_mec_has_vtx_lf = 0;
    int tot_mec_has_gvtx_lf = 0;
    int tot_mec_bvtx_lf = 0;
    int tot_other_has_vtx_lf = 0;
    int tot_other_has_gvtx_lf = 0;
    int tot_other_bvtx_lf = 0;

    int total_pass = 0;
    int total_eene_pass = 0;
    int total_fs_pass = 0;
    int total_loc_pass = 0;
    int total_pene_pass = 0;

    // values for location cut
    float xmax = 256;
    float xmin = 0;
    float ymax = 117;
    float ymin = -117;
    float zmax = 1036;
    float zmin = .3;
    float thresh = 10;

    TH2F* h2dcutg = new TH2F("h2dcutg","",20,0.,20.,30,0.,3000.);
    TH2F* h2dcutb = new TH2F("h2dcutb","",20,0.,20.,30,0.,3000.);

    TH2F* h2dbox_adc_gwo = new TH2F("h2dbox_adc_gwo","",20,0.,20.,50,0.,10000.);
    TH2F* h2dbox_adc_bwo = new TH2F("h2dbox_adc_bwo","",20,0.,20.,50,0.,10000.);
    TH2F* h2dbox_adc_gwi = new TH2F("h2dbox_adc_gwi","",20,0.,20.,50,0.,10000.);
    TH2F* h2dbox_adc_bwi = new TH2F("h2dbox_adc_bwi","",20,0.,20.,50,0.,10000.);

    int Nentries = tree->GetEntries();
    std::cout << "Nentries " << Nentries << std::endl;
    std::vector<int> bad_ev;
    std::vector<int> good_ev;
    // Nentries = 400;
    // std::vector<int> specific_entries = {3, 168, 175, 260, 282, 306, 346, 516, 519, 553, 560, 562, 601, 703, 1004, 1084, 1137, 1196, 1272, 1341, 1526, 1547, 1576, 1596, 1611, 1660, 1697, 1720, 1912, 1971, 2011, 2030, 2042, 2107, 2167, 2213, 2283, 2322, 2327, 2448, 2450, 2538, 2654, 2837, 2844, 2924, 3011, 3068, 3117, 3184, 3201, 3283, 3342, 3398, 3433, 3446, 3449, 3554, 3689, 3731, 3741, 3802, 3809, 3875, 3914, 3915, 3922, 3932, 3935, 3973, 4035, 4041, 4235, 4316, 4394, 4414, 4591, 4618, 4642, 4794, 4860, 4911, 4968, 5072, 5089, 5284, 5317, 5434, 5465, 5690, 5763, 5764, 5920, 5944, 5974, 6024, 6036, 6100, 6125, 6147, 6270, 6278, 6330, 6428, 6469, 6532, 6563, 6567, 6622, 6725, 6747, 6885, 7062, 7085, 7166, 7191, 7196, 7415, 7423, 7471, 7561, 7590, 7594, 7599, 7647, 7772, 7851, 7872, 7976, 7999, 8077, 8101, 8105, 8192, 8233, 8282, 8300, 8327, 8343, 8355, 8361, 8414, 8678, 8783, 8924, 9014, 9096, 9132, 9159, 9163, 9333, 9376, 9397, 9418, 9465, 9482, 9538};
    for(int i=0; i<Nentries; i++){
    // for (int ii=0; ii<specific_entries.size(); ii++){
    // int i = specific_entries[ii];

      bool make_before_after = false;
      num_good_vtx_lf =0;
      num_bad_vtx_lf = 0;
      // set bools for cuts on events
      bool p_energy_cut = false;
      bool e_energy_cut = false;
      bool vtx_fiducial_cut = false;
      bool fs_number_cut = false;
      // nu_energy = (float) i ;
      tree->GetEntry(i);
      if (i%100 == 0){
        std::cout<<"Entry: "<< i <<std::endl;
      }
      tot_nu +=num_numu;
      tot_nu +=num_nue;


      // p_energy_cut
      int total_highen_p = 0;
      if (p_enedep->size() > 0){
	       for (int j = 0; j<p_enedep->size();j++){
	          if(p_enedep->at(j) > 60){
	             total_highen_p ++;
	          }
	       }
      }
      if (total_highen_p == 1){
        p_energy_cut = true;
        total_pene_pass ++;
      }
      else continue;

      //e_energy_cut
      int total_highen_e = 0;
      if (e_enedep->size() > 0){
	       for (int j = 0; j<e_enedep->size();j++){
	          if(e_enedep->at(j) > 35){
	             total_highen_e ++;
	          }
	       }
      }
      if (total_highen_e == 1){
        	e_energy_cut = true;
        	total_eene_pass++;
      }
      else continue;

      // final state particles cut
      // if (num_pi0 == 0 && num_piplus ==0 && num_piminus == 0 && num_n ==0){
      if (num_pi0 == 0 && num_piplus ==0 && num_piminus == 0 ){

        	fs_number_cut = true;
        	total_fs_pass ++;
      }
      else continue;

      //vtx fiducial cut
      float x_coor = _scex->at(0);
      float y_coor = _scex->at(1);
      float z_coor = _scex->at(2);
      // std::cout << x_coor << " " << (thresh+xmin) <<std::endl;
      if(x_coor > (thresh + xmin) && x_coor < (xmax-thresh)){
        	if(y_coor > (thresh + ymin) && y_coor < (ymax-thresh)){
          	  if(z_coor > (thresh + zmin) && z_coor < (zmax-thresh)){
            	    vtx_fiducial_cut = true;
            	    total_loc_pass++;
          	  }
        	}
      }
      else continue;

      TH2F before_cuts(*hywo);
      TH2F after_cuts(*hywo);
      TH2F before("before","before",3456,0.,3456,1008,0.,1008.);
      TH2F after("after","after",3456,0.,3456,1008,0.,1008.);
      if (make_before_after){
        for (int x=0;x<3456;x++){
          for (int y=0;y<1008;y++){
            float croi = hcroi->GetBinContent(x+1,y+1);
          	//ssnetbg
          	float ssnetbg = hssnetbg->GetBinContent(x+1,y+1);
          	//early
          	float early = hearly->GetBinContent(x+1,y+1);
          	//late
          	float late = hlate->GetBinContent(x+1,y+1);
          	//dwall
          	float dwall = hdwall->GetBinContent(x+1,y+1);
            bool fails_cuts = (croi >0.1 || ssnetbg >0.3 || early>100 || late>100 || dwall>20);
            // bool unclustered = (croi==1000 || ssnetbg==1000 || early==1000 || late==1000 || dwall==1000);
            // bool passes_cuts = (croi <=0.1 && ssnetbg <=0.3 && early<=100 && late<=100 && dwall<=20);
            // bool box_dim_bool = (box_dim_needed <=7);
          	if (fails_cuts) {
              after_cuts.SetBinContent(x+1,y+1,0.0);
            }
            // else{
            //   // after_cuts.SetBinContent()
            // }

          }
        }


        before.SetMarkerColor(kBlack);
        before.SetMarkerStyle(kOpenCircle);
        before.SetMarkerSize(4);
        after.SetMarkerColor(kBlack);
        after.SetMarkerStyle(kOpenCircle);
        after.SetMarkerSize(4);
      }
      /************** DL cuts **********/
      for(int j=0; j<good_vtx_pixel->size(); j++){
        	int binx = good_vtx_pixel->at(j)[3];
        	int biny = good_vtx_pixel->at(j)[0];

        	//croi
        	float croi = hcroi->GetBinContent(binx+1,biny+1);

        	//ssnetbg
        	float ssnetbg = hssnetbg->GetBinContent(binx+1,biny+1);

        	//early
        	float early = hearly->GetBinContent(binx+1,biny+1);

        	//late
        	float late = hlate->GetBinContent(binx+1,biny+1);

        	//dwall
        	float dwall = hdwall->GetBinContent(binx+1,biny+1);

        	//adc
        	float adc[5][5];
        	float tot_adc=0;
        	float xdist=0;
        	float ydist=0;
          int box_dim_needed = 10; // larger than acutal range
          for(int jk=-2; jk<2+1; jk++ ){
          	  for(int kl=-2; kl<2+1; kl++){
                  if( (hcroi->GetBinContent(binx+1+jk,biny+1+kl) <= 0.1  )&&
                      (hssnetbg->GetBinContent(binx+1+jk,biny+1+kl) <= 0.3 ) &&
                      (hearly->GetBinContent(binx+1+jk,biny+1+kl) <= 100 ) &&
                      (hlate->GetBinContent(binx+1+jk,biny+1+kl) <= 100 ) &&
                      (hdwall->GetBinContent(binx+1+jk,biny+1+kl) <= 20 ) )
                      {
                      tot_adc += hywo->GetBinContent(binx+1+jk,biny+1+kl);
                      if(hywo->GetBinContent(binx+1+jk,biny+1+kl)>0.){
                          xdist = std::fabs(jk);
                          ydist = std::fabs(kl);
                          int box_dim =  (int)(std::max(xdist,ydist)*2+1); //*2+1 to get box dim
                          if (box_dim < box_dim_needed){box_dim_needed = box_dim;}
                      }
                  }
          	  }
        	}



          h2dcutg->Fill(box_dim_needed,tot_adc);



        	//	 std::cout << "good vtx " << j <<" binx " << binx << " biny " << biny << " croi frac " << croi << " ssnetbg " << ssnetbg
        	//<< " num early " << early << " num late " << late << " num dwall " << dwall <<" 5pix tot adc " << tot_adc << std::endl;


          bool unclustered = (croi==1000 || ssnetbg==1000 || early==1000 || late==1000 || dwall==1000);
          bool passes_cuts = (croi <=0.1 && ssnetbg <=0.3 && early<=100 && late<=100 && dwall<=20);
          bool box_dim_bool = (box_dim_needed <=7);
        	if (((unclustered == true) || (passes_cuts == true)) && box_dim_bool) {
            num_good_vtx_lf++;
          }
          if ((unclustered == true) || (passes_cuts == true)){
            for (int i=0;i<10;i++){
              int dim = i*2+1;
              // std::cout << "Dim: " << dim << std::endl;
              int dim_loop = ((dim-1)/2);
              float tot_adc_wo = 0;
              float tot_adc_wi = 0;
              for (int x=-dim_loop;x<dim_loop+1;x++){
                for (int y = -dim_loop;y<dim_loop+1;y++){
                  tot_adc_wo += hywo->GetBinContent(binx+1+x,biny+1+y);
                  if( (hcroi->GetBinContent(binx+1+x,biny+1+y) <= 0.1  )&&
                      (hssnetbg->GetBinContent(binx+1+x,biny+1+y) <= 0.3 ) &&
                      (hearly->GetBinContent(binx+1+x,biny+1+y) <= 100 ) &&
                      (hlate->GetBinContent(binx+1+x,biny+1+y) <= 100 ) &&
                      (hdwall->GetBinContent(binx+1+x,biny+1+y) <= 20 ) )
                      {
                      tot_adc_wi += hywo->GetBinContent(binx+1+x,biny+1+y);
                  }
                }
              }
              // if (tot_adc_wo > 0){
                h2dbox_adc_gwo->Fill(dim,tot_adc_wo);
              // }
              // if (tot_adc_wi > 0){
                h2dbox_adc_gwi->Fill(dim,tot_adc_wi);
              // }
            }
          } //if unclustered keep vtx
      }

      for(int j=0; j<bad_vtx_pixel->size(); j++){
        	int binx = bad_vtx_pixel->at(j)[3];
        	int biny = bad_vtx_pixel->at(j)[0];
          before.Fill(binx,biny);

        	//croi
        	float croi = hcroi->GetBinContent(binx+1,biny+1);
        	//ssnetbg
        	float ssnetbg = hssnetbg->GetBinContent(binx+1,biny+1);

        	//early
        	float early = hearly->GetBinContent(binx+1,biny+1);

        	//late
        	float late = hlate->GetBinContent(binx+1,biny+1);

        	//dwall
        	float dwall = hdwall->GetBinContent(binx+1,biny+1);

        	//adc
        	float adc[5][5];
        	float tot_adc=0;
        	float xdist=0;
        	float ydist=0;
          int box_dim_needed = 10; // larger than actual range

        	for(int jk=-2; jk<2+1; jk++ ){
          	  for(int kl=-2; kl<2+1; kl++){
                  if( (hcroi->GetBinContent(binx+1+jk,biny+1+kl) <= 0.1  )&&
                      (hssnetbg->GetBinContent(binx+1+jk,biny+1+kl) <= 0.3 ) &&
                      (hearly->GetBinContent(binx+1+jk,biny+1+kl) <= 100 ) &&
                      (hlate->GetBinContent(binx+1+jk,biny+1+kl) <= 100 ) &&
                      (hdwall->GetBinContent(binx+1+jk,biny+1+kl) <= 20 ) )
                      {
                      tot_adc += hywo->GetBinContent(binx+1+jk,biny+1+kl);
                      if(hywo->GetBinContent(binx+1+jk,biny+1+kl)>0.){
                          xdist = std::fabs(jk);
                          ydist = std::fabs(kl);
                          int box_dim =  (int)(std::max(xdist,ydist)*2+1); //*2+1 to get box dim
                          if (box_dim < box_dim_needed){box_dim_needed = box_dim;}
                      }
                  }
          	  }
        	}



        	h2dcutb->Fill(box_dim_needed,tot_adc);


          bool unclustered = (croi==1000 || ssnetbg==1000 || early==1000 || late==1000 || dwall==1000);
          bool passes_cuts = (croi <=0.1 && ssnetbg <=0.3 && early<=100 && late<=100 && dwall<=20);
          bool box_dim_bool = (box_dim_needed <=7);

        	if (((unclustered == true) || (passes_cuts == true)) && box_dim_bool) {
            num_bad_vtx_lf++;
          }

          if ((unclustered == true) || (passes_cuts == true)){
            if (make_before_after){
              after.Fill(binx,biny);
              std::cout << "Wire: "<< binx << "    Tick: " << biny << std::endl;
              std::cout << "unclustered: " << unclustered << "    passes_cuts: " << passes_cuts << std::endl;
              if (unclustered){
                std::cout << "This point was unclustered." << std::endl;
              }
              else{
                std::cout << "Croi (<=0.1) :: "<< croi << "  SSNETBG (<=0.3) :: "<< ssnetbg <<"  Early (<=100) :: "<< early <<"  Late (<=100) :: "<< late <<"  DWALL (<=20) :: "<< dwall  <<"  Box Needed (<=7) :: "<< box_dim_needed<< std::endl;
              }
            }

            for (int i=0;i<10;i++){
              int dim = i*2+1;
              // std::cout << "Dim: " << dim << std::endl;
              int dim_loop = ((dim-1)/2);
              float tot_adc_wo = 0;
              float tot_adc_wi = 0;
              for (int x=-dim_loop;x<dim_loop+1;x++){
                for (int y = -dim_loop;y<dim_loop+1;y++){
                  tot_adc_wo += hywo->GetBinContent(binx+1+x,biny+1+y);
                  if( (hcroi->GetBinContent(binx+1+x,biny+1+y) <= 0.1  )&&
                      (hssnetbg->GetBinContent(binx+1+x,biny+1+y) <= 0.3 ) &&
                      (hearly->GetBinContent(binx+1+x,biny+1+y) <= 100 ) &&
                      (hlate->GetBinContent(binx+1+x,biny+1+y) <= 100 ) &&
                      (hdwall->GetBinContent(binx+1+x,biny+1+y) <= 20 ) )
                      {
                      tot_adc_wi += hywo->GetBinContent(binx+1+x,biny+1+y);
                  }

                }
              }
              // if (tot_adc_wo > 0){
                h2dbox_adc_bwo->Fill(dim,tot_adc_wo);
              // }
              // if (tot_adc_wi > 0){
                h2dbox_adc_bwi->Fill(dim,tot_adc_wi);
              // }
            }
          } //if unclustered keep vtx
      }
      if (make_before_after){
        TH2F truevtx("vtxmarkers_true","vtxmarkers_true",3456,0.,3456,1008,0.,1008.);
        truevtx.SetMarkerColor(kBlack);
        truevtx.SetMarkerStyle(kOpenSquare);
        truevtx.SetMarkerSize(4);
        truevtx.Fill(nupixel[0][3],nupixel[0][0]);


        TCanvas can_before("can_before", "can_before ", 3456, 1008);
        // can_before.SetLeftMargin(0.2);
        // can_before.SetRightMargin(0.2);

        before_cuts.SetOption("COLZ");
        before_cuts.SetTitle("Before Cuts");
        before_cuts.Draw();
        truevtx.Draw("SAME");
        before.Draw("SAME");

        std::string str = "before_after_bad_vtx/before_cuts_"+std::to_string(i)+".png";
        char cstr[str.size() + 1];
        str.copy(cstr,str.size()+1);
        cstr[str.size()] = '\0';
        can_before.SaveAs(cstr);

        TCanvas can_after("can_after", "can_after ", 3456, 1008);
        // can_after.SetLeftMargin(0.2);
        // can_after.SetRightMargin(0.2);

        after_cuts.SetOption("COLZ");
        after_cuts.SetTitle("After Cuts");
        after_cuts.Draw();
        truevtx.Draw("SAME");
        after.Draw("SAME");


        str = "before_after_bad_vtx/after_cuts_"+std::to_string(i)+".png";
         cstr[str.size() + 1];
        str.copy(cstr,str.size()+1);
        cstr[str.size()] = '\0';
        can_after.SaveAs(cstr);
      }


      if (e_energy_cut && p_energy_cut && fs_number_cut && vtx_fiducial_cut){
          total_pass ++;
        	if (ccnc == 0 && interaction_mode == 0) tot_ccqe +=1;
        	if (ccnc == 0 && interaction_mode == 10) tot_mec +=1;
        	if (!(ccnc==0 && interaction_mode == 10) && !(ccnc==0 && interaction_mode ==0)) tot_other +=1;


        	// w/o tagger
        	if(num_good_vtx>0 || num_bad_vtx>0){
          	  tot_has_vtx ++;
          	  tot_bvtx += num_bad_vtx;
          	  if(ccnc==0 && interaction_mode==0){
            	    tot_ccqe_has_vtx ++;
            	    tot_ccqe_bvtx += num_bad_vtx;
          	  }
          	  if(ccnc==0 && interaction_mode==10){
            	    tot_mec_has_vtx ++;
            	    tot_mec_bvtx += num_bad_vtx;
          	  }
          	  if(!(ccnc==0 && interaction_mode==0) && !(ccnc==0 && interaction_mode==10)){
            	    tot_other_has_vtx ++;
            	    tot_other_bvtx += num_bad_vtx;
          	  }
        	}
        	if(num_good_vtx>0){
          	  tot_has_gvtx ++;
          	  tot_gvtx += num_good_vtx;
          	  if(ccnc==0 && interaction_mode==0) tot_ccqe_has_gvtx ++;
          	  if(ccnc==0 && interaction_mode==10) tot_mec_has_gvtx ++;
          	  if(!(ccnc==0 && interaction_mode==0) && !(ccnc==0 && interaction_mode==10)) tot_other_has_gvtx ++;
        	}

        	// w/ tagger
        	if(num_good_vtx_tagger>0 || num_bad_vtx_tagger>0){
          	  tot_has_vtx_tag ++;
          	  tot_bvtx_tag += num_bad_vtx_tagger;
          	  if(ccnc==0 && interaction_mode==0){
            	    tot_ccqe_has_vtx_tag ++;
            	    tot_ccqe_bvtx_tag += num_bad_vtx_tagger;
          	  }
          	  if(ccnc==0 && interaction_mode==10){
            	    tot_mec_has_vtx_tag ++;
            	    tot_mec_bvtx_tag += num_bad_vtx_tagger;
          	  }
          	  if(!(ccnc==0 && interaction_mode==0) && !(ccnc==0 && interaction_mode==10)){
            	    tot_other_has_vtx_tag ++;
            	    tot_other_bvtx_tag += num_bad_vtx_tagger;
          	  }

        	}
        	if(num_good_vtx_tagger>0){
          	  tot_has_gvtx_tag ++;
          	  tot_gvtx_tag += num_good_vtx_tagger;
          	  if(ccnc==0 && interaction_mode==0) tot_ccqe_has_gvtx_tag ++;
          	  if(ccnc==0 && interaction_mode==10) tot_mec_has_gvtx_tag ++;
          	  if(!(ccnc==0 && interaction_mode==0) && !(ccnc==0 && interaction_mode==10)) tot_other_has_gvtx_tag ++;

        	}
          if((num_good_vtx_lf == 0) && (num_good_vtx_tagger > 0)){
            // if (good_ev.size() < 10){
              std::cout << "Missing Good Vtx:" << good_ev.size() <<std::endl;
              good_ev.push_back(i);
              // if ((bad_ev.size() == 10) && (good_ev.size() == 10) ) {break;}
            // }
          }
          if((num_bad_vtx_tagger == 0) && (num_bad_vtx_lf > 0)){
            // if (bad_ev.size() < 10){
              std::cout << "Extra Bad Vtx:" << bad_ev.size() <<std::endl;
              bad_ev.push_back(i);
              // if ((bad_ev.size() == 10) && (good_ev.size() == 10) ) {break;}
            // }
          }
        	// w/ DL tagger
        	if(num_good_vtx_lf>0 || num_bad_vtx_lf>0){
          	  tot_has_vtx_lf ++;
          	  tot_bvtx_lf += num_bad_vtx_lf;
          	  if(ccnc==0 && interaction_mode==0){
            	    tot_ccqe_has_vtx_lf ++;
            	    tot_ccqe_bvtx_lf += num_bad_vtx_lf;
          	  }
          	  if(ccnc==0 && interaction_mode==10){
            	    tot_mec_has_vtx_lf ++;
            	    tot_mec_bvtx_lf += num_bad_vtx_lf;
          	  }
          	  if(!(ccnc==0 && interaction_mode==0) && !(ccnc==0 && interaction_mode==10)){
            	    tot_other_has_vtx_lf ++;
            	    tot_other_bvtx_lf += num_bad_vtx_lf;
          	  }
        	}
        	if(num_good_vtx_lf>0){
          	  tot_has_gvtx_lf ++;
          	  tot_gvtx_lf += num_good_vtx_lf;
          	  if(ccnc==0 && interaction_mode==0) tot_ccqe_has_gvtx_lf ++;
          	  if(ccnc==0 && interaction_mode==10) tot_mec_has_gvtx_lf ++;
          	  if(!(ccnc==0 && interaction_mode==0) && !(ccnc==0 && interaction_mode==10)) tot_other_has_gvtx_lf ++;
        	}

        	//if(num_good_vtx_tagger>0 && num_good_vtx_lf==0)
        	//  std::cout <<"Entry "<< i <<" "<< run <<" "<< subrun <<" "<< event <<" has tagger good vtx, no DL good vtx" << std::endl;
      }

  }
  std::cout << "Good Ev Vector:" <<std::endl;
  for (int jj=0;jj<good_ev.size();jj++){
    std::cout << good_ev.at(jj) << ", ";
  }
  std::cout << std::endl;
  std::cout << "Bad Ev Vector:" <<std::endl;
  for (int jj=0;jj<bad_ev.size();jj++){
    std::cout << bad_ev.at(jj) << ", ";
  }
    TCanvas cane("can", "histograms ", 1200, 800);
    cane.SetLeftMargin(0.2);
    cane.SetRightMargin(0.2);
    TH2F h2dbox_adc_b_g_ratio_wo = (*h2dbox_adc_bwo)/(*h2dbox_adc_gwo);
    h2dbox_adc_b_g_ratio_wo.SetOption("COLZ");
    h2dbox_adc_b_g_ratio_wo.SetTitle("Radio Original ADC, Bad/Good");
    h2dbox_adc_b_g_ratio_wo.SetXTitle("Box Size (AxA)");
    h2dbox_adc_b_g_ratio_wo.SetYTitle("Charge Total");
    h2dbox_adc_b_g_ratio_wo.Draw();
    // cane.SaveAs("ratio_dlcut_wo_b_g_5.png");

    TCanvas canf("can", "histograms ", 1200, 800);
    canf.SetLeftMargin(0.2);
    canf.SetRightMargin(0.2);
    TH2F h2dbox_adc_b_g_ratio_wi = (*h2dbox_adc_bwi)/(*h2dbox_adc_gwi);
    h2dbox_adc_b_g_ratio_wi.SetOption("COLZ");
    h2dbox_adc_b_g_ratio_wi.SetTitle("Radio DL REMOVED ADC, Bad/Good");
    h2dbox_adc_b_g_ratio_wi.SetXTitle("Box Size (AxA)");
    h2dbox_adc_b_g_ratio_wi.SetYTitle("Charge Total");
    h2dbox_adc_b_g_ratio_wi.Draw();
    // canf.SaveAs("ratio_dlcut_wi_b_g_5.png");

    TH1D  *projbwiX_ratio = h2dbox_adc_b_g_ratio_wi.ProjectionX();
    projbwiX_ratio->SetXTitle("Box Size (AxA)");
    projbwiX_ratio->SetYTitle("Count");
    TCanvas can_bwiX_rat("can_bwoX","can_bwoX", 1200, 800);
    can_bwiX_rat.SetLeftMargin(0.2);
    can_bwiX_rat.SetRightMargin(0.2);
    projbwiX_ratio->Draw();
    // can_bwiX_rat.SaveAs("projx_bbox_ratio_wi_5.png");

    TCanvas cana("can", "histograms ", 1200, 800);
    cana.SetLeftMargin(0.2);
    cana.SetRightMargin(0.2);
    h2dbox_adc_gwo->SetOption("COLZ");
    h2dbox_adc_gwo->SetTitle("Good Vtx Original ADC");
    h2dbox_adc_gwo->SetXTitle("Box Size (AxA)");
    h2dbox_adc_gwo->SetYTitle("Charge Total");
    h2dbox_adc_gwo->Draw();
    // cana.SaveAs("h2dbox_adc_gwo_post_dlcut_5.png");

    TH1D  *projgwoX = h2dbox_adc_gwo->ProjectionX();
    projgwoX->SetXTitle("Box Size (AxA)");
    projgwoX->SetYTitle("Count");
    TCanvas can_gwoX("can_gwoX","can_gwoX", 1200, 800);
    can_gwoX.SetLeftMargin(0.2);
    can_gwoX.SetRightMargin(0.2);
    projgwoX->Draw();
    // can_gwoX.SaveAs("box_adc_gwoX_post_dlcut_5.png");

    TCanvas canb("can", "histograms ", 1200, 800);
    canb.SetLeftMargin(0.2);
    canb.SetRightMargin(0.2);
    h2dbox_adc_bwo->SetOption("COLZ");
    h2dbox_adc_bwo->SetTitle("Bad Vtx Original ADC");
    h2dbox_adc_bwo->SetXTitle("Box Size (AxA)");
    h2dbox_adc_bwo->SetYTitle("Charge Total");
    h2dbox_adc_bwo->Draw();
    // canb.SaveAs("h2dbox_adc_bwo_post_dlcut_5.png");

    TH1D  *projbwoX = h2dbox_adc_bwo->ProjectionX();
    projbwoX->SetXTitle("Box Size (AxA)");
    projbwoX->SetYTitle("Count");
    TCanvas can_bwoX("can_bwoX","can_bwoX", 1200, 800);
    can_bwoX.SetLeftMargin(0.2);
    can_bwoX.SetRightMargin(0.2);
    projbwoX->Draw();
    // can_bwoX.SaveAs("box_adc_bwoX_post_dlcut_5.png");

    TCanvas canc("can", "histograms ", 1200, 800);
    canc.SetLeftMargin(0.2);
    canc.SetRightMargin(0.2);
    h2dbox_adc_bwi->SetTitle("Bad Vtx DL Cut ADC");
    h2dbox_adc_bwi->SetXTitle("Box Size (AxA)");
    h2dbox_adc_bwi->SetYTitle("Charge Total");
    h2dbox_adc_bwi->SetOption("COLZ");
    h2dbox_adc_bwi->Draw();
    // canc.SaveAs("h2dbox_adc_bwi_post_dlcut_5.png");

    TCanvas cand("can", "histograms ", 1200, 800);
    cand.SetLeftMargin(0.2);
    cand.SetRightMargin(0.2);
    h2dbox_adc_gwi->SetTitle("Good Vtx DL Cut ADC");
    h2dbox_adc_gwi->SetXTitle("Box Size (AxA)");
    h2dbox_adc_gwi->SetYTitle("Charge Total");
    h2dbox_adc_gwi->SetOption("COLZ");
    h2dbox_adc_gwi->Draw();
    // cand.SaveAs("h2dbox_adc_gwi_post_dlcut_5.png");

    TCanvas* can1  = new TCanvas("can1","",500,500);
    can1->cd();
    h2dcutg->Draw("colz");
    can1->SaveAs("2Dcut_good_5.root");
    TCanvas* can2  = new TCanvas("can2","",500,500);
    can2->cd();
    h2dcutb->Draw("colz");
    // can2->SaveAs("2Dcut_bad_5.root");

    //std::cout<<"pass p energy cut: "<<total_pene_pass<<std::endl;
    //std::cout<<"pass e energy cut: "<<total_eene_pass<<std::endl;
    //std::cout<<"pass final state cut: "<<total_fs_pass<<std::endl;
    //std::cout<<"pass fiducial cut: "<<total_loc_pass<<std::endl;
    std::cout << "pass all cuts: "<< total_pass << std::endl;
    std::cout << "=======================================================" <<std::endl;
    std::cout << "Has reco vtx w/o tagger: " << tot_has_vtx << std::endl;
    std::cout << "Has reco vtx w/i tagger: " << tot_has_vtx_tag << std::endl;
    std::cout << "Has reco vtx w/i DL: " << tot_has_vtx_lf << std::endl;
    std::cout << "Has good reco vtx w/o tagger: " << tot_has_gvtx << std::endl;
    std::cout << "Has good reco vtx w/i tagger: " << tot_has_gvtx_tag << std::endl;
    std::cout << "Has good reco vtx w/i DL: " << tot_has_gvtx_lf << std::endl;
    std::cout << "Good reco vtx w/o tagger: " << tot_gvtx << std::endl;
    std::cout << "Good reco vtx w/i tagger: " << tot_gvtx_tag << std::endl;
    std::cout << "Good reco vtx w/i DL: " << tot_gvtx_lf << std::endl;
    std::cout << "Bad reco vtx w/o tagger: " << tot_bvtx << std::endl;
    std::cout << "Bad reco vtx w/i tagger: " << tot_bvtx_tag << std::endl;
    std::cout << "Bad reco vtx w/i DL: " << tot_bvtx_lf << std::endl;
    //std::cout << "Rejected good vtx w/i tagger: " << tot_gvtx - tot_gvtx_tag << std::endl;
    //std::cout << "Rejected bad vtx w/i tagger: " << tot_bvtx - tot_bvtx_tag << std::endl;
    //std::cout << "Rejected good vtx w/i DL: " << tot_gvtx - tot_gvtx_lf << std::endl;
    //std::cout << "Rejected bad vtx w/i DL: " << tot_bvtx - tot_bvtx_lf << std::endl;
    std::cout << "=======================================================" <<std::endl;
    std::cout << "=========================== CCQE ======================" <<std::endl;
    std::cout << "Has reco vtx w/o tagger: " << tot_ccqe_has_vtx << std::endl;
    std::cout << "Has reco vtx w/i tagger: " << tot_ccqe_has_vtx_tag << std::endl;
    std::cout << "Has reco vtx w/i DL: " << tot_ccqe_has_vtx_lf << std::endl;
    std::cout << "Has good reco vtx w/o tagger: " << tot_ccqe_has_gvtx << std::endl;
    std::cout << "Has good reco vtx w/i tagger: " << tot_ccqe_has_gvtx_tag << std::endl;
    std::cout << "Has good reco vtx w/i DL: " << tot_ccqe_has_gvtx_lf << std::endl;
    //std::cout << "Good reco vtx w/o tagger: " << tot_ccqe_gvtx << std::endl;
    //std::cout << "Good reco vtx w/i tagger: " << tot_ccqe_gvtx_tag << std::endl;
    //std::cout << "Good reco vtx w/i DL: " << tot_ccqe_gvtx_lf << std::endl;
    std::cout << "Bad reco vtx w/o tagger: " << tot_ccqe_bvtx << std::endl;
    std::cout << "Bad reco vtx w/i tagger: " << tot_ccqe_bvtx_tag << std::endl;
    std::cout << "Bad reco vtx w/i DL: " << tot_ccqe_bvtx_lf << std::endl;
    std::cout << "=========================== MEC ======================" <<std::endl;
    std::cout << "Has reco vtx w/o tagger: " << tot_mec_has_vtx << std::endl;
    std::cout << "Has reco vtx w/i tagger: " << tot_mec_has_vtx_tag << std::endl;
    std::cout << "Has reco vtx w/i DL: " << tot_mec_has_vtx_lf << std::endl;
    std::cout << "Has good reco vtx w/o tagger: " << tot_mec_has_gvtx << std::endl;
    std::cout << "Has good reco vtx w/i tagger: " << tot_mec_has_gvtx_tag << std::endl;
    std::cout << "Has good reco vtx w/i DL: " << tot_mec_has_gvtx_lf << std::endl;
    //std::cout << "Good reco vtx w/o tagger: " << tot_mec_gvtx << std::endl;
    //std::cout << "Good reco vtx w/i tagger: " << tot_mec_gvtx_tag << std::endl;
    //std::cout << "Good reco vtx w/i DL: " << tot_mec_gvtx_lf << std::endl;
    std::cout << "Bad reco vtx w/o tagger: " << tot_mec_bvtx << std::endl;
    std::cout << "Bad reco vtx w/i tagger: " << tot_mec_bvtx_tag << std::endl;
    std::cout << "Bad reco vtx w/i DL: " << tot_mec_bvtx_lf << std::endl;
    std::cout << "=========================== Other ======================" <<std::endl;
    std::cout << "Has reco vtx w/o tagger: " << tot_other_has_vtx << std::endl;
    std::cout << "Has reco vtx w/i tagger: " << tot_other_has_vtx_tag << std::endl;
    std::cout << "Has reco vtx w/i DL: " << tot_other_has_vtx_lf << std::endl;
    std::cout << "Has good reco vtx w/o tagger: " << tot_other_has_gvtx << std::endl;
    std::cout << "Has good reco vtx w/i tagger: " << tot_other_has_gvtx_tag << std::endl;
    std::cout << "Has good reco vtx w/i DL: " << tot_other_has_gvtx_lf << std::endl;
    //std::cout << "Good reco vtx w/o tagger: " << tot_other_gvtx << std::endl;
    //std::cout << "Good reco vtx w/i tagger: " << tot_other_gvtx_tag << std::endl;
    //std::cout << "Good reco vtx w/i DL: " << tot_other_gvtx_lf << std::endl;
    std::cout << "Bad reco vtx w/o tagger: " << tot_other_bvtx << std::endl;
    std::cout << "Bad reco vtx w/i tagger: " << tot_other_bvtx_tag << std::endl;
    std::cout << "Bad reco vtx w/i DL: " << tot_other_bvtx_lf << std::endl;


     /*
    // =========================================================================
    // EVENT DISPLAY CODE
    tree->GetEntry(2766);
    hywo->SetOption("COLZ");
    int size1 = 3;
    int size2 = 4;
    std::cout << "Interacion mode: " << interaction_mode << std::endl;
    std::cout << "Num bad vtx: " << bad_vtx_pixel->size() << std::endl;
    std::cout << "Num Good vtx tagger: " << good_vtx_tag_pixel->size() << std::endl;
    std::cout << "Num Good vtx no tagger: " << good_vtx_pixel->size() << std::endl;

    TH2F* truevtx = new TH2F("vtxmarkers_true","vtxmarkers_true",3456,0.,3456,1008,0.,1008.);
    truevtx->Fill(scenupixel[0][3],scenupixel[0][0]);
    float truecol = scenupixel[0][3];
    float truerow = scenupixel[0][0];
    int zoomcmax = truecol+15;
    int zoomrmax = truerow+15;
    int zoomcmin = truecol-15;
    int zoomrmin = truerow-15;
    std::cout << "row: "<<truerow <<" col: "<<truecol<<std::endl;
    truevtx->SetMarkerColor(kBlack);
    truevtx->SetMarkerStyle(kOpenSquare);
    truevtx->SetMarkerSize(4);

    TH2F vtxmarkers_good_no_tag("vtxmarkers_good_no_tag","vtxmarkers_good_no_tag",3456,0.,3456,1008,0.,1008.);
      for (int i = 0; i<good_vtx_pixel->size(); i++){
        std::vector<int> vtxvec_wo =good_vtx_pixel->at(i);
        int row = vtxvec_wo[0];
        int col = vtxvec_wo[3];
        std::cout << row << " "<< col << std::endl;
        vtxmarkers_good_no_tag.Fill(col,row);
      }
    TH2F vtxmarkers_bad_no_tag("vtxmarkers_bad_no_tag","vtxmarkers_bad_no_tag",3456,0.,3456,1008,0.,1008.);
    for (int i = 0; i<bad_vtx_pixel->size(); i++){
      std::vector<int> vtxvec_wo =bad_vtx_pixel->at(i);
      int row = vtxvec_wo[0];
      int col = vtxvec_wo[3];
      vtxmarkers_bad_no_tag.Fill(col,row);
    }

    TH2F vtxmarkers_good_old_tagger("vtxmarkers_good_old_tagger","vtxmarkers_good_old_tagger",3456,0.,3456,1008,0.,1008.);
    for (int i = 0; i<good_vtx_tag_pixel->size(); i++){
      std::vector<int> vtxvec_wo =good_vtx_tag_pixel->at(i);
      int row = vtxvec_wo[0];
      int col = vtxvec_wo[3];
      vtxmarkers_good_old_tagger.Fill(col,row);
    }

    TH2F vtxmarkers_bad_old_tagger("vtxmarkers_bad_old_tagger","vtxmarkers_bad_old_tagger",3456,0.,3456,1008,0.,1008.);
    for (int i = 0; i<bad_vtx_tag_pixel->size(); i++){
      std::vector<int> vtxvec_wo =bad_vtx_tag_pixel->at(i);
      int row = vtxvec_wo[0];
      int col = vtxvec_wo[3];
      vtxmarkers_bad_old_tagger.Fill(col,row);
    }

    TH2F vtxmarkers_good_dl("vtxmarkers_good_dl","vtxmarkers_good_dl",3456,0.,3456,1008,0.,1008.);
    for (int i = 0; i<good_vtx_lf_pixel->size(); i++){
      std::vector<int> vtxvec_wi =good_vtx_lf_pixel->at(i);
      int row = vtxvec_wi[0];
      int col = vtxvec_wi[3];
      vtxmarkers_good_dl.Fill(col,row);
    }

    TH2F vtxmarkers_bad_dl("vtxmarkers_bad_dl","vtxmarkers_bad_dl",3456,0.,3456,1008,0.,1008.);
    for (int i = 0; i<bad_vtx_lf_pixel->size(); i++){
      std::vector<float> vtxvec_wi =bad_vtx_lf_pixel->at(i);
      float row = vtxvec_wi[0];
      float col = vtxvec_wi[3];
      vtxmarkers_bad_dl.Fill(col,row);
    }

    //Set Marker Formats
    vtxmarkers_good_no_tag.SetMarkerColor(kBlack);
    vtxmarkers_good_no_tag.SetMarkerStyle(kOpenCross);
    vtxmarkers_good_no_tag.SetMarkerSize(4);
    vtxmarkers_bad_no_tag.SetMarkerColor(kBlack);
    vtxmarkers_bad_no_tag.SetMarkerStyle(kOpenTriangleUp);
    vtxmarkers_bad_no_tag.SetMarkerSize(4);

    vtxmarkers_good_old_tagger.SetMarkerColor(kMagenta );
    vtxmarkers_good_old_tagger.SetMarkerStyle(kOpenCross);
    vtxmarkers_good_old_tagger.SetMarkerSize(size1);
    vtxmarkers_bad_old_tagger.SetMarkerColor(kMagenta );
    vtxmarkers_bad_old_tagger.SetMarkerStyle(kOpenTriangleUp);
    vtxmarkers_bad_old_tagger.SetMarkerSize(size1);

    vtxmarkers_good_dl.SetMarkerColor(kRed );
    vtxmarkers_good_dl.SetMarkerStyle(kOpenCross);
    vtxmarkers_good_dl.SetMarkerSize(size1);
    vtxmarkers_bad_dl.SetMarkerColor(kRed);
    vtxmarkers_bad_dl.SetMarkerStyle(kOpenTriangleUp);
    vtxmarkers_bad_dl.SetMarkerSize(size1);

    TCanvas* c3 = new TCanvas("c3","Y plane without tagger - zoom",1000, 600);
    hywo->GetXaxis()->SetRange(zoomcmin,zoomcmax);
    hywo->GetYaxis()->SetRange(zoomrmin,zoomrmax);
    hywo->Draw();
    truevtx->Draw("SAME");
    vtxmarkers_good_no_tag.Draw("SAME");
    vtxmarkers_bad_no_tag.Draw("SAME");
    vtxmarkers_good_old_tagger.Draw("SAME");
    vtxmarkers_bad_old_tagger.Draw("SAME");
    c3->SaveAs(("hywo_zoom_2766.root"));

    TCanvas* c4 = new TCanvas("c4","Y plane with tagger - zoom",1000, 600);
    hywi->GetXaxis()->SetRange(zoomcmin,zoomcmax);
    hywi->GetYaxis()->SetRange(zoomrmin,zoomrmax);
    hywi->Draw();
    truevtx->Draw("SAME");
    vtxmarkers_bad_no_tag.Draw("SAME");
    vtxmarkers_good_no_tag.Draw("SAME");
    vtxmarkers_good_dl.Draw("SAME");
    vtxmarkers_bad_dl.Draw("SAME");
    c4->SaveAs(("hywi_zoom_2766.png"));
    */
  }
