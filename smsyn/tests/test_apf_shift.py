"""Test smsyn.inst.hires.shift.shift code by using the ganymede
spectrum and a synthesized coelho model
"""
from astropy.io import fits
import os.path
from matplotlib import pylab as plt
import numpy as np
from argparse import ArgumentParser

import smsyn.inst.apf
from smsyn.inst.apf.loadspec import read_fits
import smsyn.inst.apf.shift
import smsyn.library
import smsyn.io.spectrum

if __name__=="__main__":
    psr = ArgumentParser()
    psr.add_argument('libraryfile',type=str,help="path to library hdf file")
    args = psr.parse_args()
    lib = smsyn.library.read_hdf(args.libraryfile, wavlim=[4000,7000])
    dirname = os.path.dirname(os.path.abspath(smsyn.__file__))

    orders = range(30,56)
    
    wscale = fits.open(os.path.join(dirname,"inst","apf","apf_wave_bj2.fits"))
    wav = wscale[0].data[orders,:] # guess at wavelength scale

    flux,uflux = read_fits('/mir4/iodfits/rabh.225.fits', flatten=True, order=6, geterr=True, verbose=True, specorders=orders)
    header = fits.getheader('/mir4/iodfits/rabh.225.fits')
    header['name'] = header['OBJECT']
    header['obs'] = 'rabh.225'
        
    ref_wav = np.logspace(np.log10(wav[0,0]),np.log10(wav[-1,-1]),64000)
    ref_flux = lib.synth(ref_wav, 5777, 4.4, 0.0, 2, 0.5)
    flux_shift, uflux_shift = smsyn.inst.apf.shift.shift(

        wav, flux, uflux, ref_wav, ref_flux
    )


    spec = smsyn.io.spectrum.Spectrum(ref_wav, flux_shift, uflux_shift, header)
    spec.to_fits('/Users/bfulton/Dropbox/fulton_petigura/MOON_rabh.225.fits')
    
    plt.plot(ref_wav, ref_flux, label="Reference spectrum")
    plt.plot(ref_wav, flux_shift, label="Observed spectrum")
    plt.draw()
    plt.show()
