#from __future__ import print_function
import os,sys
import argparse

parser = argparse.ArgumentParser(description="Combine LARLITE rootfiles")
parser.add_argument("filelist",type=str,nargs='+')
parser.add_argument("-o","--output",required=True,type=str)

args = parser.parse_args(sys.argv[1:])

import ROOT
from larlite import larlite

io = larlite.storage_manager(larlite.storage_manager.kBOTH)
io.set_verbosity(0)
for f in args.filelist:
    print("merging file: ",f)
    io.add_in_filename( f )
print("merged larlite output file: ",args.output)
io.set_out_filename( args.output )
io.open()

nentries = io.get_entries()
io.next_event()
for ientry in range(nentries):
    io.next_event()

io.close()
