from __future__ import print_function
import os,sys
import ROOT as rt
from larcv import larcv
from larflow import larflow


# set input file name. this should be a ROOT file containing the 'KPSRecoManagerTree'.
# To check if your file has this tree you can type the commands:
# > root [yourfile.root]
# > .ls
# You should see:
#  KEY: TTree	KPSRecoManagerTree;1	Ana Output of KPSRecoManager algorithms
#
fname = "kpsreco_mcc9_v29e_run3_G1_extbnb_merged_dlana_ffddd628-f39a-4db4-a797-7dc323ddc0c6_kpsrecomanagerana.root"
#fname = "../../testdata/mcc9_v29e_run3_G1_extbnb/kpsreco_mcc9_v29e_run3_G1_extbnb_merged_dlana_ffddd628-f39a-4db4-a797-7dc323ddc0c6_kpsrecomanagerana.root"

# set output rootfile name
outfname = "out_example.root"

# open ROOT file
rootfile = rt.TFile( fname, "open" )

# get the tree with the data
kpsrecotree = rootfile.Get("KPSRecoManagerTree")

# get number of entries in the tree. corresponds to number of events
nentries = kpsrecotree.GetEntries()

# define the output rootfile, set to overwrite past files
outroot = rt.TFile( outfname, "recreate" )

# define a histogram
htracks = rt.TH1D("htracks","number of tracks",10,0,10)

# loop through the entries
for ientry in range(nentries):
    
    print("[ENTRY %d]"%(ientry))

    # load the entry
    nbytes = kpsrecotree.GetEntry(ientry)
    if nbytes==0:
        # if zero, out of entries, or something wrong
        break

    # each entry will have a vector of candidate vertices
    vertex_v = kpsrecotree.nufitted_v

    # get the number of vertices
    nvertices = vertex_v.size()
    print("  number of vertices in the entry: ",nvertices)

    # loop over the candidates
    for ivertex in range(nvertices):
        vtx = vertex_v[ivertex]

        ntracks  = vtx.track_v.size()
        nshowers = vtx.shower_v.size()

        # add number of tracks in this vertex into the histogram
        print("  (vertex %d) ntracks=%d"%(ivertex,ntracks))
        htracks.Fill( ntracks )

print("done with loop")

# make sure the output file is active
outroot.cd()

# write histogram to file
htracks.Write()

# close
outroot.Close()



print("""This script will have made a ROOT file with the 'htracks' histgram stored in it.
To see the histogram, open the ROOT file via:

root out_example.root

then draw the histogram via

htracks->Draw()
""")
