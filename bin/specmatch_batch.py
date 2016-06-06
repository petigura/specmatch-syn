from argparse import ArgumentParser
import smsyn.specmatch
import smsyn.smio
shelp = """
Thin wrapper around SpecMatch.coelho_match

Evaluates chi2 of observed spectrum with respect to coelho
library. Saves output to h5 data structure. Uses smio.cps_resolve to
resolve output file name. Can also specify directly.

"""

def main(args):
    obs = args.obs
    h5path = smsyn.smio.cps_resolve(obs,'h5')
    np = 8
    smsyn.specmatch.coelho_match(
        obs, h5path=h5path, debug=args.debug, numpro=np
    )

if __name__=="__main__":
    psr = ArgumentParser(description=shelp)
    psr.add_argument(
        'obs',type=str,help='CPS ID. Output h5file from smio.cps_resolve'
    )
    psr.add_argument('--snr',type=int,default=-1)

    psr.add_argument(
        '--debug',action='store_true', 
        help='Run over a trimmed grid for fast debuging'
    )
    psr.add_argument(
        '--np',type=int,default=1,help='Number of processors to use'
    )
    args  = psr.parse_args()
    main(args)
