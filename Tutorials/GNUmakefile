CXX=g++

CXXFLAGS = -g -fPIC
INCLUDES =`root-config --cflags`
INCLUDES += `larcv-config --includes`
INCLUDES += `larlite-config --includes`

LIBS = `root-config --libs`
LIBS += `larlite-config --libs`
LIBS += `larcv-config --libs`

all: make_event_display

make_event_display: make_event_display.cxx
				$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o make_event_display $(LIBS)
