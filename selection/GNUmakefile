CXX=g++

CXXFLAGS = -g -fPIC
INCLUDES =`root-config --cflags`
INCLUDES += `larcv-config --includes`
INCLUDES += `larlite-config --includes`

LIBS = `root-config --libs`
LIBS += `larlite-config --libs`
LIBS += `larcv-config --libs`

all: dev_cosmictag

dev_cosmictag: dev_cosmictag.cxx
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o dev_cosmictag $(LIBS)
