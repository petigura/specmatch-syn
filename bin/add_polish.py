from argparse import ArgumentParser
from time import strftime

import pandas as pd

from smsyn import results
from smsyn import h5plus
from smsyn import smio
from smsyn import conf
from smsyn import telluric

from matplotlib.pylab import *
rc('savefig',dpi=200)
import h5py

def main(args):

    obs = args.obs
    targd = smio.kbc_query(obs)
    targd['type'] = 'cps'

    if args.i!=None:
        h5fileinp = args.i
    else:
        h5fileinp = smio.cps_resolve(obs,'h5')

    if args.o!=None:
        h5fileout = args.o
    else:
        h5fileout = h5fileinp


    if args.scheme=='fm':
        print "adding forward modeling parameters to %s" % h5fileout

        wloL = conf.wloL_lc
        d=[]
        for wlo in wloL:
            d +=[results.getpars_lincomb(h5fileinp,wlo,usemask=False)]
        par0 = pd.DataFrame(d).groupby('obs').mean().iloc[0]
        par0['vsini'] = 5

        with h5plus.File(h5fileout) as h5:    
            
            psfsig = h5.attrs['sig']
            deckname = h5.attrs['DECKNAME']
            par0['psfsig'] = telluric.psf_prior(psfsig,deckname)
            d = results.getpars_polish(targd,par0,h5=h5,usemask=True,snr=args.snr)
            g = h5.create_group('fm')
            h5plus.dict_to_attrs(g,d)
            g.attrs['polish_stop_time'] = strftime("%Y-%m-%d %H:%M:%S")
            g.attrs['polish_sha'] = smio.get_repo_head_sha()

if __name__=="__main__":
    psr = ArgumentParser()
    psr.add_argument('obs',type=str,help='ID of observation')
    psr.add_argument(
        'scheme',type=str,help='Method to produce fit choose [lc|fm]'
    )
    psr.add_argument('--snr',type=int,help='Desired SNR level')
    psr.add_argument('--wlo',type=int,help='wlo')
    psr.add_argument('-i',type=str,help='Input h5file')
    psr.add_argument('-o',type=str,help='Output h5file')
    args  = psr.parse_args()
    main(args)
