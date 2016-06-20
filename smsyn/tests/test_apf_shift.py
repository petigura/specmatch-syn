"""Test smsyn.inst.hires.shift.shift code by using the moon
spectrum and a synthesized coelho model
"""
import os
from argparse import ArgumentParser

from astropy.io import fits
import numpy as np

import smsyn.inst.hires.shift
import smsyn.inst.apf.loadspec
import smsyn.library
import smsyn.io.spectrum
from smsyn import DATA_DIR

if __name__=="__main__":
    name = 'MOON'
    obs = 'rabh.225'
    inpfile = os.path.join(DATA_DIR,'{}_{}.fits'.format(name,obs))
    outfile = '{}_{}.sm.fits'.format(name,obs)
    
    psr = ArgumentParser()
    psr.add_argument('libfile',type=str,help="path to library hdf file")

    args = psr.parse_args()
    lib = smsyn.library.read_hdf(args.libfile, wavlim=[4000,7000])
    dirname = os.path.dirname(os.path.abspath(smsyn.__file__))
    hduL = fits.open(inpfile)

    orders = range(30,51)
    
    wav = fits.getdata(os.path.join(DATA_DIR, 'apf_wave_bj2.fits')) # guess at wavelength scale
    wav = wav[orders,:]
    
    flux,uflux = smsyn.inst.apf.loadspec.read_fits(inpfile, flatten=True, geterr=True, specorders=orders)

    
    ref_wav = np.logspace(np.log10(wav[0,0]),np.log10(wav[-1,-1]),64000)
    ref_flux = lib.synth(ref_wav, 5700, 4.4, 0.0, 2, 2)
    flux_shift, uflux_shift = smsyn.inst.hires.shift.shift(
        wav, flux, uflux, ref_wav, ref_flux
    )

    spec = smsyn.io.spectrum.Spectrum(
        ref_wav, flux_shift, uflux_shift, header=dict(name=name,obs=obs)
    )
    spec.to_fits(outfile) 
