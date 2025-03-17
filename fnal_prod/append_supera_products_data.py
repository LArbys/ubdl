import sys
import ROOT as rt

mergefile = sys.argv[1]
larcvfile = sys.argv[2]

import ROOT as rt

flarcv = rt.TFile(larcvfile)
larcvtrees = ["chstatus_wire_tree", "image2d_thrumu_tree", "image2d_wire_tree"]

trees = {}

for t in larcvtrees:
    trees[t] = flarcv.Get(t)

fmerge = rt.TFile(mergefile,"update")

for t in larcvtrees:
    trees[t].CloneTree().Write("",rt.TObject.kOverwrite)

flarcv.Close()
fmerge.Close()

