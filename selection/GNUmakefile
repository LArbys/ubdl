CXX=g++

CXXFLAGS = -g -fPIC
INCLUDES =`root-config --cflags`
INCLUDES += `larcv-config --includes`
INCLUDES += `larlite-config --includes`
INCLUDES += -I$(UBLARCVAPP_INCDIR)

LIBS = `root-config --libs`
LIBS += `larlite-config --libs`
LIBS += `larcv-config --libs`
LIBS += -L$(UBLARCVAPP_LIBDIR)

all: dev_cosmictag vertex_eval

dev_cosmictag: dev_cosmictag.cxx
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o dev_cosmictag $(LIBS)

vertex_eval: vertex_eval.cxx
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o vertex_eval $(LIBS)
