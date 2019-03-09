import os,sys,time,logging,argparse
from UBSSNetClient import UBSSNetClient


if __name__ == "__main__":


    logging.basicConfig(filename="client.log",level=logging.DEBUG)
    
    # endpoint:
    endpoint  = "tcp://localhost:8080"

    input_dir = "/home/twongjirad/working/larbys/ubdl/testdata/ex1"
    larcv_supera_file   = input_dir+"/supera-Run000001-SubRun006867.root"
    larlite_opreco_file = input_dir+"/opreco-Run000001-SubRun006867.root"
    output_larcv_filename = "out_ubssnet_test.root"

    client = UBSSNetClient(endpoint,larcv_supera_file,"wire",
                            output_larcv_filename,
                            larlite_opreco_file=larlite_opreco_file,
                            apply_opflash_roi=True, tick_backwards=True)
    client.connect()


    client.process_entries(0,3)

    client.finalize()

    print "[ENTER] to quit."
    raw_input()
