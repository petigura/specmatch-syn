"""Test smsyn.inst.hires.shift.shift code by using the ganymede
spectrum and a synthesized coelho model
"""
from astropy.io import fits
import os.path
from matplotlib import pylab as plt
import numpy as np
from argparse import ArgumentParser

import smsyn.inst.hires.shift
import smsyn.library
if __name__=="__main__":
    psr = ArgumentParser()
    psr.add_argument('libraryfile',type=str,help="path to library hdf file")
    args = psr.parse_args()
    lib = smsyn.library.read_hdf(args.libraryfile, wavlim=[4000,7000])
    dirname = os.path.dirname(os.path.abspath(smsyn.__file__))
    hduL = fits.open(os.path.join(dirname,'data/GANYMEDE_rj76.279.fits'))

    wav = hduL[2].data # guess at wavelength scale
    flux = hduL[0].data 
    flux /= np.median(flux,axis=1)[:,np.newaxis] # normalize order by order
    uflux = hduL[1].data # uncertanties

    ref_wav = np.logspace(np.log10(wav[0,0]),np.log10(wav[-1,-1]),64000)
    ref_flux = lib.synth(ref_wav, 5700, 4.4, 0.0, 2, 2)
    flux_shift, uflux_shift = smsyn.inst.hires.shift.shift(

        wav, flux, uflux, ref_wav, ref_flux
    )

    plt.plot(ref_wav, ref_flux, label="Reference spectrum")
    plt.plot(ref_wav, flux_shift, label="Observed spectrum")
    plt.draw()
    plt.show()
