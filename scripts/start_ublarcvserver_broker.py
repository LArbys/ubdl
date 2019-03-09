#!/usr/bin/env python
import os,sys,argparse,logging
from ublarcvserver import Broker

parser = argparse.ArgumentParser()
parser.add_argument("port",type=int,help="port the broker will listen on")
parser.add_argument("-l","--logfile",type=str, default=None,
                    help="where the log file is writen to")
parser.add_argument("-d","--debug",action="store_true",
                    help="set logger level to debug")

if __name__ == "__main__":

    args = parser.parse_args()


    level = logging.INFO
    if args.debug:
        level = logging.DEBUG

    if args.logfile is not None:
        logging.basicConfig(filename=args.logfile,level=level)

    log = logging.getLogger("main")
    bindpoint="tcp://*:%d"%(args.port)
    log.info("Starting the broker at {}".format(bindpoint))

    broker = Broker(bind=bindpoint)
    broker.run()
    
    log.info("Broker finished")
    
