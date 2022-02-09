import os,sys,argparse

# define arguments

parser = argparse.ArgumentParser(description='Dump some meta-data.')
parser.add_argument('inputfiles',type=str, nargs='+',help='list of input files to run')
parser.add_argument('-o','--output',type=str,required=True,help="name of output file")
                    
args = parser.parse_args()

# check if output file already exists
if os.path.exists(args.output):
    print("The output file already exists. Stopping to prevent overwriting.")
    sys.exit(0)

# import our software repo libraries
import ROOT as rt
from larlite import larlite
from ublarcvapp import ublarcvapp

# SETUP THE INPUT FILE
io = larlite.storage_manager( larlite.storage_manager.kREAD )
for f in args.inputfiles:
    f = f.strip()
    if not os.path.exists(f):
        raise ValueError("Did not find input file: ",f)
    io.add_in_filename(f)
io.open()
nentries = io.get_entries()
print("Number of entries in the file")

# SETUP THE OUTPUT FILE
out = rt.TFile(args.output,"new")

# in this example, we will make a couple of histograms
#  -- of the true neutrino energy
#  -- of the visible energy in the event

# define histograms
hnu = rt.TH1D("hnuE","true neutrino energy; MeV; counts",50,0,5000) # 100 MeV bins, 0 to 5 GeV
hevis = rt.TH1D("hevis","Visible energy; MeV; counts",50,0,5000)
hnu_vs_evis = rt.TH2D("hnuE_v_evis",";MeV;MeV",50,0,5000,50,0,5000)

for i in range(nentries):
    print("[ ENTRY ",i," ] =======================")
    io.go_to(i)

    ev_mctruth_v = io.get_data( larlite.data.kMCTruth, "generator" )
    mctruth = ev_mctruth_v.at(0)
    nu_energy = mctruth.GetNeutrino().Nu().Momentum().E()
    ccnc = mctruth.GetNeutrino().CCNC()
    interactiontype = mctruth.GetNeutrino().InteractionType()
    print("  neutrino energy: %.3f"%(nu_energy)," GeV")
    print("  CC or NC: ","CC" if ccnc==0 else "NC")
    print("  interaction type: ",interactiontype)

    # fill the true neutrino energy
    hnu.Fill( nu_energy*1000.0 )

    # calculate the visible energy
    # defined as the kinetic energy of particles
    # we will loop over simulated particles, picking out only those from the neutrino interaction (origin=1)
    evis = 0.0
    
    # scan mc tracks
    ev_tracks = io.get_data( larlite.data.kMCTrack, "mcreco" )
    for itrack in range(ev_tracks.size()):
        track = ev_tracks.at(itrack)

        # beside the origin, we only want to consider primaries, i.e. the intitial particles from the nu interaction
        # one way to identify primaries is to ask if the track id and its mother id is the same
        isprimary = track.TrackID()==track.MotherTrackID()
        
        if isprimary and track.Origin()==1:
            # neutrino origin
            p = track.Start().Momentum() # 4-momentum at start of track
            mass = p.Mag() # sqrt of the norm -- the mass
            E = p.E() # the energy component
            ke = E-mass # KE
            evis += ke
    # scan mc showers
    ev_showers = io.get_data( larlite.data.kMCShower, "mcreco" )
    for ishower in range(ev_showers.size()):
        shower = ev_showers.at(ishower)

        isprimary = shower.TrackID()==shower.MotherTrackID()
        
        if isprimary and shower.Origin()==1:
            # neutrino origin
            p = shower.Start().Momentum() # 4-momentum at start of shower
            mass = p.Mag() # sqrt of the norm -- the mass
            E = p.E() # the energy component
            ke = E-mass # KE
            evis += ke
    hevis.Fill( evis )

    # we can make the joint distribution
    hnu_vs_evis.Fill( nu_energy*1000.0, evis )

    

io.close()

# write the file with the histograms in it
out.Write()
