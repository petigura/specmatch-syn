from argparse import ArgumentParser

import h5py
from matplotlib.pylab import plt
from time import strftime

from smsyn import telluric
from smsyn import smio


shelp = """
Determine the PSF from telluric lines.


1. Read in obs from h5file
2. Measure telluric linewidth
3. Write PSF width as attribute of h5file
4. Option to create a diagnostic plot

"""


def main(args):
    files = []
    for obs in args.obs:
        files+=[smio.cps_resolve(obs,'h5')]

    for f,obs in zip(files,args.obs):
        try:
            with h5py.File(f,'r+') as h5:
                if args.plot:
                    fig,ax = plt.subplots(figsize=(12,4))

                sig = telluric.telluric_psf(obs,plot=args.plot)
                h5.attrs['sig'] = sig
                h5.attrs['telluric_stop_time'] = strftime("%Y-%m-%d %H:%M:%S")
                h5.attrs['telluric_sha'] = smio.get_repo_head_sha()

                if args.plot:
                    plotdir = smio.cps_resolve(obs,'plotdir')
                    nameobs = smio.cps_resolve(obs,'nameobs')
                    pngfile = '%s/%s_tell.png' % (plotdir,nameobs)

                    plt.gcf().savefig(pngfile)
                    print "created %s" % pngfile
                    plt.close('all')

            print "%s %.2f" % (obs,sig)
        except (KeyError,IOError):
            print "telluric_psf.py: failed %s" % f



if __name__=="__main__":
    psr = ArgumentParser(description='Thin wrapper around fitspec')
    psr.add_argument(
        'obs',nargs='+',type=str, help='CPS ID(s) to process'
    )
    psr.add_argument(
        '--files',nargs='+',type=str, help='files to process'
    )

    psr.add_argument(
        '--plot',action='store_true',help='create diag plot?'
    )
    args  = psr.parse_args()
    main(args)
