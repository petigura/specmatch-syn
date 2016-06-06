from argparse import ArgumentParser
import smsyn.smio
import smsyn.restwav
import glob
#from conf import *

def main(args):
    obsL = args.obs
    force = args.f
    obstype = 'db'
    for obs in obsL:
        path = smsyn.smio.cps_resolve(obs,'restwav')
        if (len(glob.glob(path))==0) | force:
            smsyn.restwav.restwav(obs)
        else:
            if force:
                os.remove(path) 
                smsyn.restwav.restwav(obs)
            else:
                print "skipping %s" % obs

if __name__=="__main__":
    psr = ArgumentParser(description='Thin wrapper around fitspec')
    psr.add_argument('--obs',type=str,nargs='+',help='ID of observation')
    psr.add_argument(
        '-f', action='store_true', default=False, help='overwrite existing?'
    )
    args  = psr.parse_args()
    main(args)
