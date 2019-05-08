#include <iostream>
#include <map>
#include <utility>
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
#include "DataFormat/mctrack.h"
#include "DataFormat/mcshower.h"
#include "LArUtil/LArProperties.h"
#include "LArUtil/Geometry.h"
#include "LArUtil/ClockConstants.h"
#include "LArUtil/SpaceChargeMicroBooNE.h"
#include "LArUtil/DetectorProperties.h"

// larcv
#include "larcv/core/DataFormat/IOManager.h"
#include "larcv/core/DataFormat/EventImage2D.h"
#include "larcv/core/DataFormat/EventChStatus.h"
#include "larcv/core/DataFormat/EventClusterMask.h"
#include "larcv/core/DataFormat/EventPGraph.h"
#include "larcv/core/DataFormat/EventROI.h"

// ublarcv
#include "ublarcvapp/UBWireTool/UBWireTool.h"

std::vector<int> getProjectedPixel(const std::vector<float>& pos3d,
				   const larcv::ImageMeta& meta,
				   const int nplanes,
				   const float fracpixborder=1.5 );

int main( int nargs, char** argv ) {
  //std::string ubmrcnn = argv[1];
  std::string vtx = argv[1];
  std::string vtx2 = argv[2];
  std::string ubclust = argv[3];
  std::string supera = argv[4];
  std::string larcvtruth = argv[5];


  // ADC 
  larcv::IOManager io_larcv( larcv::IOManager::kREAD, "", larcv::IOManager::kTickBackward );
  io_larcv.add_in_file( supera );
  io_larcv.initialize();

  // True nu vtx from PartROI && instance
  larcv::IOManager io_larcvtruth( larcv::IOManager::kREAD, "", larcv::IOManager::kTickBackward );
  io_larcvtruth.add_in_file( larcvtruth );
  io_larcvtruth.initialize();

  // Reco vtx from PGraph (w/out tagger)
  larcv::IOManager io_vtx( larcv::IOManager::kREAD, "", larcv::IOManager::kTickBackward );
  io_vtx.add_in_file( vtx );
  io_vtx.initialize();
  
  // Reco vtx from PGraph (w/ tagger)
  larcv::IOManager io_vtx2( larcv::IOManager::kREAD, "", larcv::IOManager::kTickBackward );
  io_vtx2.add_in_file( vtx2 );
  io_vtx2.initialize();

  // Remaining Larflow3dhits
  larlite::storage_manager io_ubclust( larlite::storage_manager::kREAD );
  io_ubclust.add_in_filename( ubclust );
  io_ubclust.open();

  //larcv::IOManager io_ubmrcnn( larcv::IOManager::kREAD, "", larcv::IOManager::kTickBackward );
  //io_ubmrcnn.add_in_file( ubmrcnn );
  //io_ubmrcnn.initialize();

  // Ana tree
  
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

  std::vector<std::vector<float> > good_vtx;
  std::vector<std::vector<float> > bad_vtx;
  std::vector<std::vector<float> > good_vtx_tag;
  std::vector<std::vector<float> > bad_vtx_tag;
  std::vector<std::vector<float> > good_vtx_lf;
  std::vector<std::vector<float> > bad_vtx_lf;
  
  std::vector<std::vector<int> > good_vtx_pixel;
  std::vector<std::vector<int> > bad_vtx_pixel;
  std::vector<std::vector<int> > good_vtx_lf_pixel;
  std::vector<std::vector<int> > bad_vtx_lf_pixel;
  std::vector<std::vector<int> > good_vtx_tag_pixel;
  std::vector<std::vector<int> > bad_vtx_tag_pixel;
  
  TFile* fout = new TFile("cosmictag_vertexana.root","recreate");
  TTree* tree = NULL;
  bool saveTree=true;
  if(saveTree){
    tree = new TTree("tree","event tree");
    tree->Branch("run",&run,"run/I");
    tree->Branch("subrun",&subrun,"subrun/I");
    tree->Branch("event",&event,"event/I");
    tree->Branch("num_nu",&num_nu,"num_nu/I");
    tree->Branch("num_good_vtx",&num_good_vtx,"num_good_vtx/I");
    tree->Branch("num_bad_vtx",&num_bad_vtx,"num_bad_vtx/I");
    tree->Branch("num_good_vtx_tagger",&num_good_vtx_tagger,"num_good_vtx_tagger/I");
    tree->Branch("num_bad_vtx_tagger",&num_bad_vtx_tagger,"num_bad_vtx_tagger/I");
    tree->Branch("num_good_vtx_lf",&num_good_vtx_lf,"num_good_vtx_lf/I");
    tree->Branch("num_bad_vtx_lf",&num_bad_vtx_lf,"num_bad_vtx_lf/I");
    tree->Branch("_scex",&_scex);
    tree->Branch("nupixel",&nupixel);
    //tree->Branch("good_vtx_pixel", &good_vtx_pixel);
    //tree->Branch("bad_vtx_pixel", &bad_vtx_pixel);
  }

  ::larutil::SpaceChargeMicroBooNE* sce = new ::larutil::SpaceChargeMicroBooNE;
  
  int nentries = io_larcv.get_n_entries();
  //nentries = 10; //temp
  for (int i=0; i<nentries; i++) {

    io_larcv.read_entry(i);
    io_larcvtruth.read_entry(i);
    io_vtx.read_entry(i);
    io_vtx2.read_entry(i);
    io_ubclust.go_to(i);


    auto ev_cluster        = (larlite::event_larflowcluster*)io_ubclust.get_data(larlite::data::kLArFlowCluster,  "cosmictag" );
    auto ev_img            = (larcv::EventImage2D*)io_larcv.get_data( larcv::kProductImage2D, "wire" );
    auto ev_instance       = (larcv::EventImage2D*)io_larcvtruth.get_data( larcv::kProductImage2D, "instance" );
    auto ev_partroi        = (larcv::EventROI*)(io_larcvtruth.get_data( larcv::kProductROI,"segment"));
    auto ev_pgraph         = (larcv::EventPGraph*) io_vtx.get_data( larcv::kProductPGraph,"test");
    auto ev_pgraph2        = (larcv::EventPGraph*) io_vtx2.get_data( larcv::kProductPGraph,"test");
    
    run    = io_larcv.event_id().run();
    subrun = io_larcv.event_id().subrun();
    event  = io_larcv.event_id().event();
    std::cout <<"run "<< run <<" subrun " << subrun <<" event " << event << std::endl;

    auto& wire_img = ev_img->Image2DArray();
    auto& wire_meta = wire_img.at(2).meta(); 
    
    float _tx=0;
    float _ty=0;
    float _tz=0;
    float _tt=0;
    std::vector<float> _x(3,0);
    std::vector<int> pixel;
    _scex.clear();
    nupixel.clear();
    good_vtx.clear();
    bad_vtx.clear();
    good_vtx_tag.clear();
    bad_vtx_tag.clear();
    good_vtx_lf.clear();
    bad_vtx_lf.clear();
    good_vtx_pixel.clear();
    bad_vtx_pixel.clear();
    good_vtx_tag_pixel.clear();
    bad_vtx_tag_pixel.clear();
    good_vtx_lf_pixel.clear();
    bad_vtx_lf_pixel.clear();
    
    for(auto const& roi : ev_partroi->ROIArray()){
      if(std::abs(roi.PdgCode()) == 12 || std::abs(roi.PdgCode()) == 14) {
	_tx = roi.X();
	_ty = roi.Y();
	_tz = roi.Z();
	_tt = roi.T();
	auto const offset = sce->GetPosOffsets(_tx,_ty,_tz);
	_scex[0] = _tx - offset[0] + 0.7;
	_scex[1] = _ty + offset[1];
	_scex[2] = _tz + offset[2];
	//nupixel = ublarcvapp::UBWireTool::getProjectedImagePixel(pos, wire_meta, 3);
	nupixel = getProjectedPixel(_scex, wire_meta, 3);

      }      
    }
    //std::cout <<"nu vtx "<< _scex[0] << " "<< _scex[1] << " "<< _scex[2] << std::endl;
    
    for(size_t pgraph_id = 0; pgraph_id < ev_pgraph->PGraphArray().size(); ++pgraph_id) {
      auto const& pgraph = ev_pgraph->PGraphArray().at(pgraph_id);
      _x[0] = pgraph.ParticleArray().front().X();
      _x[1] = pgraph.ParticleArray().front().Y();
      _x[2] = pgraph.ParticleArray().front().Z();
      pixel = getProjectedPixel(_x, wire_meta, 3);
      //std::cout << _x[0] <<" "<< _x[1] <<" "<< _x[2] <<" "<< pixel[0] <<" "<< pixel[3] << std::endl;
      if(std::sqrt(std::pow(_x[0]-_scex[0],2)+std::pow(_x[1]-_scex[1],2)+std::pow(_x[2]-_scex[2],2))<=5.0){
	good_vtx.push_back(_x);
	good_vtx_pixel.push_back(pixel);
      }
      else{bad_vtx.push_back(_x);
	bad_vtx_pixel.push_back(pixel);
      }
    }
    for(size_t pgraph_id = 0; pgraph_id < ev_pgraph2->PGraphArray().size(); ++pgraph_id) {
      auto const& pgraph = ev_pgraph2->PGraphArray().at(pgraph_id);
      _x[0] = pgraph.ParticleArray().front().X();
      _x[1] = pgraph.ParticleArray().front().Y();
      _x[2] = pgraph.ParticleArray().front().Z();
      pixel = getProjectedPixel(_x, wire_meta, 3);
      //std::cout << _x[0] <<" "<< _x[1] <<" "<< _x[2] << std::endl;
      if(std::sqrt(std::pow(_x[0]-_scex[0],2)+std::pow(_x[1]-_scex[1],2)+std::pow(_x[2]-_scex[2],2))<=5.0){
	good_vtx_tag.push_back(_x);
	good_vtx_tag_pixel.push_back(pixel);
      }else{
	bad_vtx_tag.push_back(_x);
	bad_vtx_tag_pixel.push_back(pixel);
      }
    }

    // fill remaining ADC
    larcv::Image2D remaining = wire_img.at(2);
    remaining.paint(0.0);
    for(unsigned int i=0; i<ev_cluster->size(); i++){
      for(auto& hit : ev_cluster->at(i)){
	int tick = hit.tick; //time tick
	int wire = hit.srcwire; // Y wire
	  
	if ( wire_meta.min_x()<=wire && wire<wire_meta.max_x()
	     && wire_meta.min_y()<=tick && tick<wire_meta.max_y() ) {
	  int col = wire_meta.col( wire );
	  int row = wire_meta.row( tick );
	  remaining.set_pixel(row,col,wire_img.at(2).pixel(row,col));
	}
      }
    }
    // now check if we rejected any vertices
    // we keep vertices that are w/in 3 pix from a larflow3dhit
    for(int k=0; k<good_vtx_pixel.size(); k++){
      int row = good_vtx_pixel.at(k)[0];
      int col = good_vtx_pixel.at(k)[3];
      if(remaining.pixel(row,col)>0){
	good_vtx_lf.push_back(good_vtx.at(k));
	good_vtx_lf_pixel.push_back(good_vtx_pixel.at(k));
      }
      else if(remaining.pixel(row,col-1)>0){
	good_vtx_lf.push_back(good_vtx.at(k));
	good_vtx_lf_pixel.push_back(good_vtx_pixel.at(k));
      }
      else if(remaining.pixel(row,col+1)>0){
	good_vtx_lf.push_back(good_vtx.at(k));
	good_vtx_lf_pixel.push_back(good_vtx_pixel.at(k));
      }

      else if(remaining.pixel(row-1,col-1)>0){
	good_vtx_lf.push_back(good_vtx.at(k));
	good_vtx_lf_pixel.push_back(good_vtx_pixel.at(k));
      }
      else if(remaining.pixel(row-1,col)>0){
	good_vtx_lf.push_back(good_vtx.at(k));
	good_vtx_lf_pixel.push_back(good_vtx_pixel.at(k));
      }
      else if(remaining.pixel(row-1,col+1)>0){
	good_vtx_lf.push_back(good_vtx.at(k));
	good_vtx_lf_pixel.push_back(good_vtx_pixel.at(k));
      }
      else if(remaining.pixel(row+1,col-1)>0){
	good_vtx_lf.push_back(good_vtx.at(k));
	good_vtx_lf_pixel.push_back(good_vtx_pixel.at(k));
      }
      else if(remaining.pixel(row+1,col)>0){
	good_vtx_lf.push_back(good_vtx.at(k));
	good_vtx_lf_pixel.push_back(good_vtx_pixel.at(k));
      }
      else if(remaining.pixel(row+1,col+1)>0){
	good_vtx_lf.push_back(good_vtx.at(k));
	good_vtx_lf_pixel.push_back(good_vtx_pixel.at(k));
      }
    }

    for(int k=0; k<bad_vtx_pixel.size(); k++){
      int row = bad_vtx_pixel.at(k)[0];
      int col = bad_vtx_pixel.at(k)[3];
      if(remaining.pixel(row,col)>0){
	bad_vtx_lf.push_back(bad_vtx.at(k));
	bad_vtx_lf_pixel.push_back(bad_vtx_pixel.at(k));
      }
      else if(remaining.pixel(row,col-1)>0){
	bad_vtx_lf.push_back(bad_vtx.at(k));
	bad_vtx_lf_pixel.push_back(bad_vtx_pixel.at(k));
      }
      else if(remaining.pixel(row,col+1)>0){
	bad_vtx_lf.push_back(bad_vtx.at(k));
	bad_vtx_lf_pixel.push_back(bad_vtx_pixel.at(k));
      }

      else if(remaining.pixel(row-1,col-1)>0){
	bad_vtx_lf.push_back(bad_vtx.at(k));
	bad_vtx_lf_pixel.push_back(bad_vtx_pixel.at(k));
      }
      else if(remaining.pixel(row-1,col)>0){
	bad_vtx_lf.push_back(bad_vtx.at(k));
	bad_vtx_lf_pixel.push_back(bad_vtx_pixel.at(k));
      }
      else if(remaining.pixel(row-1,col+1)>0){
	bad_vtx_lf.push_back(bad_vtx.at(k));
	bad_vtx_lf_pixel.push_back(bad_vtx_pixel.at(k));
      }
      else if(remaining.pixel(row+1,col-1)>0){
	bad_vtx_lf.push_back(bad_vtx.at(k));
	bad_vtx_lf_pixel.push_back(bad_vtx_pixel.at(k));
      }
      else if(remaining.pixel(row+1,col)>0){
	bad_vtx_lf.push_back(bad_vtx.at(k));
	bad_vtx_lf_pixel.push_back(bad_vtx_pixel.at(k));
      }
      else if(remaining.pixel(row+1,col+1)>0){
	bad_vtx_lf.push_back(bad_vtx.at(k));
	bad_vtx_lf_pixel.push_back(bad_vtx_pixel.at(k));
      }

    }
      
    /*    
    std::cout << "num good vtx " << good_vtx.size()  << std::endl;
    std::cout << "num bad vtx " << bad_vtx.size() << std::endl;
    std::cout << "num good vtx tag " << good_vtx_tag.size() << std::endl;
    std::cout << "num bad vtx tag " << bad_vtx_tag.size() << std::endl;
    std::cout << "num good vtx lf" << good_vtx_lf.size() << std::endl;
    std::cout << "num bad vtx lf" << bad_vtx_lf.size()  << std::endl;
    */
    num_nu =1;
    num_good_vtx = good_vtx.size();
    num_bad_vtx = bad_vtx.size();
    num_good_vtx_tagger = good_vtx_tag.size();
    num_bad_vtx_tagger = bad_vtx_tag.size();
    num_good_vtx_lf = good_vtx_lf.size();
    num_bad_vtx_lf = bad_vtx_lf.size();

    //std::cout << _scex[1] << std::endl;
    tree->Fill();
    
  }

  fout->cd();
  tree->Write();
  fout->Close();
  
  return 0;
}

std::vector<int> getProjectedPixel( const std::vector<float>& pos3d,
				    const larcv::ImageMeta& meta,
				    const int nplanes,
				    const float fracpixborder ) {
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
}  
