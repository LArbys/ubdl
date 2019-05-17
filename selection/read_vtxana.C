
using namespace std;

TFile* fin;
TTree* tree;
int run=-1;
int subrun=-1;
int event=-1;

int num_nu;
int num_good_vtx;
int num_bad_vtx;
int num_good_vtx_tagger;
int num_bad_vtx_tagger;
int num_good_vtx_lf;
int num_bad_vtx_lf;
std::vector<float> _scex(3,0);
std::vector<int> nupixel; //tick u v y wire


void read_vtxana(){  
  fin = new TFile("cosmictag_vertexana_hadd.root","read");
  tree = (TTree*)fin->Get("tree");
  tree->SetBranchAddress("run",&run);
  tree->SetBranchAddress("subrun",&subrun);
  tree->SetBranchAddress("event",&event);
  tree->SetBranchAddress("num_nu",&num_nu);
  tree->SetBranchAddress("num_good_vtx",&num_good_vtx);
  tree->SetBranchAddress("num_bad_vtx",&num_bad_vtx);
  tree->SetBranchAddress("num_good_vtx_tagger",&num_good_vtx_tagger);
  tree->SetBranchAddress("num_bad_vtx_tagger",&num_bad_vtx_tagger);
  tree->SetBranchAddress("num_good_vtx_lf",&num_good_vtx_lf);
  tree->SetBranchAddress("num_bad_vtx_lf",&num_bad_vtx_lf);
  //tree->SetBranchAddress("_scex",&_scex);
  //tree->SetBranchAddress("nupixel",&nupixel);

  int tot_nu=0;
  int tot_has_vtx=0;
  int tot_has_vtx_tag=0;
  int tot_has_vtx_lf=0;
  int tot_good_vtx=0;
  int tot_bad_vtx=0;
  int tot_good_tag=0;
  int tot_bad_tag=0;
  int tot_good_lf=0;
  int tot_bad_lf=0;

  for(int i=0; i<tree->GetEntries(); i++){
    tree->GetEntry(i);
    tot_nu +=num_nu;
    if(num_good_vtx>0 || num_bad_vtx>0) tot_has_vtx ++;
    tot_good_vtx += num_good_vtx;
    tot_bad_vtx += num_bad_vtx;
    if(num_good_vtx_tagger>0 || num_bad_vtx_tagger>0) tot_has_vtx_tag ++;
    tot_good_tag += num_good_vtx_tagger;
    tot_bad_tag += num_bad_vtx_tagger;
    if(num_good_vtx_lf>0 || num_bad_vtx_lf>0) tot_has_vtx_lf ++;
    tot_good_lf += num_good_vtx_lf;
    tot_bad_lf += num_bad_vtx_lf;

  }

  cout << "Tot Nu: " << tot_nu << endl;
  cout << "Has reco vtx w/o tagger: " << tot_has_vtx << endl;
  cout << "Has reco vtx w/i tagger: " << tot_has_vtx_tag << endl;
  cout << "Has reco vtx w/i DL: " << tot_has_vtx_lf << endl;
  cout << "Good reco vtx w/o tagger: " << tot_good_vtx << endl;
  cout << "Good reco vtx w/i tagger: " << tot_good_tag << endl;
  cout << "Good reco vtx w/i DL: " << tot_good_lf << endl;
  cout << "Bad reco vtx w/o tagger: " << tot_bad_vtx << endl;
  cout << "Bad reco vtx w/i tagger: " << tot_bad_tag << endl;
  cout << "Bad reco vtx w/i DL: " << tot_bad_lf << endl;
  cout << "Rejected good vtx w/i tagger: " << tot_good_vtx - tot_good_tag << endl;
  cout << "Rejected bad vtx w/i tagger: " << tot_bad_vtx - tot_bad_tag << endl;
  cout << "Rejected good vtx w/i DL: " << tot_good_vtx - tot_good_lf << endl;
  cout << "Rejected bad vtx w/i DL: " << tot_bad_vtx - tot_bad_lf << endl;

}
