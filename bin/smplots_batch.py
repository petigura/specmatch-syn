from argparse import ArgumentParser

from smsyn import smplots
from smsyn import smio

import h5py
import pandas as pd
import os
from matplotlib import pylab as plt

def savefig(func):
    """
    Save figure decorating function

    1. Checks if we want to run plotting function
    2. Saves it to specified path
    3. Closes figure
    """
    def wrapper():
        if args.__getattribute__(func.__name__):
            figpath = func()
            plt.gcf().savefig(figpath)
            print "created %s" % figpath
    return wrapper

def main(args):
    for obs in args.obs:
        f = smio.cps_resolve(obs,'h5')
        plotdir = smio.cps_resolve(obs,'plotdir')
        nameobs = smio.cps_resolve(obs,'nameobs')

        # If library keyword is set, load the library parameters
        libpar = None

        with h5py.File(f) as h5:
            g = h5['6100']

            @savefig
            def panels():
                smres = g['smres'][:]
                smres = pd.DataFrame(smres)
                smres['chi'] = smres.fchi
                smplots.panels(smres,libpar=libpar)
                return '%s/%s_panels.png' % (plotdir,nameobs)

            @savefig
            def matches_chi():
                smplots.plot_matches_group(g,how='chi')
                return  '%s/%s_matches-chi.png' % (plotdir,nameobs)

            @savefig
            def quicklook():
                smplots.plot_quicklook(f)
                return  '%s/%s_quicklook.png' % (plotdir,nameobs)

            # Actually run the plotting program 
            panels()
            matches_chi()
            quicklook()

            wloL = [5200,5360,6100,5530,6210] 
            for wlo in wloL:
                @savefig
                def polish():
                    smplots.plot_polish(f,wlo,libpar=libpar)
                    return  '%s/%s_polish_%i.png' % (plotdir,nameobs,wlo)

                @savefig
                def lincomb():
                    results.getpars_lincomb(f,wlo,usemask=True,plot=True)
                    return '%s/%s_lincomb_%i.png' % (plotdir,nameobs,wlo)

                lincomb()
                polish()


if __name__=="__main__":
    psr = ArgumentParser(
        description='Thin wrapper around fitspec'
    )
    psr.add_argument(
        'obs',type=str,nargs='+', help='CPS ID(s) for stars'
    )

    psr.add_argument(
        '-i',action='store_true', help='Interactive mode. Do not save file'
    )
    psr.add_argument(
        '--panels',action='store_true', help='Generate panels plot?'
    )

    psr.add_argument(
        '--matches-chi',action='store_true', help='Generate matches plot'
    )

    psr.add_argument(
        '--lincomb',action='store_true', help='lincomb plots'
    )

    psr.add_argument(
        '--polish',action='store_true', help='set to create polish plots'
    )

    psr.add_argument(
        '--quicklook',action='store_true', help='set to create polish plots'
    )

    args  = psr.parse_args()
    main(args)
