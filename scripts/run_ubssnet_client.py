import os,sys,time,logging,argparse

parser = argparse.ArgumentParser()
parser.add_argument("brokeraddr",type=str,help="Broker Address")
parser.add_argument("supera",type=str,help="input larcv supera file")
parser.add_argument("opreco",type=str,help="input larlite opreco file")
parser.add_argument("-o","--output",type=str,required=True,help="output file")
parser.add_argument("-l","--logfile",type=str, default=None,
                    help="where the log file is writen to")
parser.add_argument("-d","--debug",action="store_true",
                    help="set logger level to debug")
parser.add_argument("-m","--mode",type=str,default="cuda",
                    help="run with device. either 'cuda' or 'cpu'")
parser.add_argument("-b","--batch-size",type=int,default=1,
                    help="batch size for each worker")
parser.add_argument("-n","--num_workers",type=int,default=1,
                    help="number of workers to launch")
parser.add_arguemnt("-t","--ssh-tunnel",type=str,
                    help="Tunnel using SSH through the given address")
parser.add_argument("-u","--ssh-user",type=str,default=None,
                    help="username for ssh tunnel command")
parser.add_argument("-a","--adc",type=str,default="wire",
                    help="ADC image producer name")
parser.add_argument("-p","--opflash",type=str,default="simpleFlashBeam",
                    help="opflash producer name")
parser.add_argument("-r","--tick-backwards",action="store_true",
                    help="larcv images stored in tick-backwards format. "+
                         "typical for older files.")



args = parser.parse_args()

from UBSSNetClient import UBSSNetClient


if __name__ == "__main__":

    level = logging.INFO
    if args.debug:
        level = logging.DEBUG
    if args.logfile is not None:
        logging.basicConfig(level=level)
    else:
        logging.basicConfig(filename=args.logfile,level=level)

    log = logging.getLogger("runclientmain")
    endpoint  = args.brokeraddr
    larcv_supera_file   = args.supera
    larlite_opreco_file = args.opreco
    if not os.path.exists(larcv_supera_file):
        log.error("given supera could not be found: "+larcv_supera_file)
        sys.exit(1)
    if not os.path.exists(larlite_opreco_file):
        log.error("given opreco could not be found: "+larlite_opreco_file)
        sys.exit(1)

    output_larcv_filename = args.output

    if args.ssh_tunnel is not None:
        if args.ssh_user is None:
            raise ValueError("If using ssh tunnel, must provide user")
        print "Using ssh, please provide password"
        ssh_password =  getpass.getpass()
        ssh_url = "%s@%s"%(args.ssh_user,args.ssh_tunnel)
    else:
        ssh_url = None
        ssh_password = None

    client = UBSSNetClient(endpoint,larcv_supera_file,
                            output_larcv_filename,
                            larlite_opreco_file=larlite_opreco_file,
                            apply_opflash_roi=True,
                            adc_producer=args.adc,
                            opflash_producer=args.opflash,
                            tick_backwards=args.tick_backwards,
                            ssh_thru_server=ssh_url,
                            ssh_password=ssh_password)
    client.connect()


    client.process_entries()

    client.finalize()

    print "[ENTER] to quit."
    raw_input()
