CXX=g++

CXXFLAGS = -g -fPIC
INCLUDES =`root-config --cflags`
INCLUDES += `larcv-config --includes`
INCLUDES += `larlite-config --includes`

LIBS = `root-config --libs`
LIBS += `larlite-config --libs`
LIBS += `larcv-config --libs`

all: make_event_display make_event_display_nbrnd create_larcvImage2d create_nbkrnd_larcvImage2d \
	GetTruthInfo make_event_display_no_background make_event_display_from_sparse \
	create_sparse_nbkrnd
	
create_sparse_nbkrnd: create_sparse_nbkrnd.cxx
				$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o create_sparse_nbkrnd $(LIBS)

make_event_display: make_event_display.cxx
				$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o make_event_display $(LIBS)

make_event_display_nbrnd: make_event_display_no_background.cxx
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o make_event_display_no_background $(LIBS)
	
create_larcvImage2d: create_larcvImage2d.cxx
				$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o create_larcvImage2d $(LIBS)
				
create_img2d_w_vertex: create_img2d_w_vertex.cxx
				$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o create_img2d_w_vertex $(LIBS)

create_nbkrnd_larcvImage2d: create_nbkrnd_larcvImage2d.cxx
				$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o create_nbkrnd_larcvImage2d $(LIBS)
	
GetTruthInfo: GetTruthInfo.cxx
				$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o GetTruthInfo $(LIBS)

make_event_display_no_background: make_event_display_no_background.cxx
				$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o make_event_display_no_background $(LIBS)

make_event_display_from_sparse: make_event_display_from_sparse.cxx
				$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o make_event_display_from_sparse $(LIBS)