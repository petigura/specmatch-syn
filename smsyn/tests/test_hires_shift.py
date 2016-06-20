"""Test smsyn.inst.hires.shift.shift code by using the ganymede
spectrum and a synthesized coelho model
"""
import os
from argparse import ArgumentParser

from astropy.io import fits
import numpy as np

import smsyn.inst.hires.shift
import smsyn.library
import smsyn.io.spectrum
from smsyn import DATA_DIR

if __name__=="__main__":
    name = 'GANYMEDE'
    obs = 'rj76.279'
    inpfile = os.path.join(DATA_DIR,'{}_{}.fits'.format(name,obs))
    outfile = '{}_{}.sm.fits'.format(name,obs)
    
    psr = ArgumentParser()
    psr.add_argument('libfile',type=str,help="path to library hdf file")

    args = psr.parse_args()
    lib = smsyn.library.read_hdf(args.libfile, wavlim=[4000,7000])
    dirname = os.path.dirname(os.path.abspath(smsyn.__file__))
    hduL = fits.open(inpfile)

    wav = hduL[2].data # guess at wavelength scale
    flux = hduL[0].data 
    flux /= np.median(flux,axis=1)[:,np.newaxis] # normalize order by order
    uflux = hduL[1].data # uncertanties

    ref_wav = np.logspace(np.log10(wav[0,0]),np.log10(wav[-1,-1]),64000)
    ref_flux = lib.synth(ref_wav, 5700, 4.4, 0.0, 2, 2)
    flux_shift, uflux_shift = smsyn.inst.hires.shift.shift(
        wav, flux, uflux, ref_wav, ref_flux
    )

    spec = smsyn.io.spectrum.Spectrum(
        ref_wav, flux_shift, uflux_shift, header=dict(name=name,obs=obs)
    )
    spec.to_fits(outfile) 
