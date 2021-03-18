CXX=g++

CXXFLAGS = -g -fPIC
INCLUDES =`root-config --cflags`
INCLUDES += `larcv-config --includes`
INCLUDES += `larlite-config --includes`

LIBS = `root-config --libs`
LIBS += `larlite-config --libs`
LIBS += `larcv-config --libs`

all: make_event_display create_larcvImage2d GetTruthInfo

make_event_display: make_event_display.cxx
				$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o make_event_display $(LIBS)

create_larcvImage2d: create_larcvImage2d.cxx
				$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o create_larcvImage2d $(LIBS)

GetTruthInfo: GetTruthInfo.cxx
				$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o GetTruthInfo $(LIBS)
