import itertools
import os

import h5py
import numpy as np

import smsyn.smio
#import smsyn.specmatch


def coelho_synth(teff,logg,fe,vsini,psf,wlo=None,whi=None,ord=None,obtype='cps',wav=None, macro=True, libfull=False):
    """
    Synthesize Coelho Model

    For a given teff, logg, fe, vsini, psf, compute a model spectrum by:

       1. Determine the 8 coelho models surounding the (teff,logg,fe)
       2. Perform trilinear interpolation
       3. Broaden with rot-macro turbulence
       4. Broaden with PSF (assume gaussian)

    Parameters
    ----------
    teff  : effective temp (K)
    logg  : surface gravity (logg)
    fe    : metalicity [Fe/H] (dex)
    vsini : rotational velocity (km/s)
    psf   : sigma for instrumental profile (pixels)
    wlo   : lower wavelength
    whi   : upper wavelength
    ord   : order

    Returns
    -------
    spec : synthesized model (record array with s, serr, w fields)

    History 
    -------
    Mar-2014 EAP created
    """
    
    tspec = smsyn.smio.getspec_h5(type='cps',obs='rj76.279',wlo=wlo,whi=whi,ord=ord)
    wav = tspec['w']


    lib = smsyn.smio.loadlibrary('/Users/petigura/Research/SpecMatch/library/library_coelho.csv')
    lib['row'] = range(len(lib))

    lib['dteff'] = np.abs(lib.teff-teff)
    lib['dlogg'] = np.abs(lib.logg-logg)
    lib['dfe'] = np.abs(lib.fe-fe)

    teff1,teff2 = lib.sort_values(by='dteff')['teff'].drop_duplicates()[:2]
    logg1,logg2 = lib.sort_values(by='dlogg')['logg'].drop_duplicates()[:2]
    fe1,fe2 = lib.sort_values(by='dfe')['fe'].drop_duplicates()[:2]

    corners = itertools.product([teff1,teff2],[logg1,logg2],[fe1,fe2])

    def getcorner(c):
        return getmodelseg(lib.ix[c],wav)
#    getcorner = lambda c : getmodelseg(lib.ix[c],w)
    c = np.vstack( map(getcorner,corners) )

    serr = c[0]['serr']
    c = c['s']
    
    v0 = [teff1, logg1, fe1]
    v1 = [teff2, logg2, fe2]
    vi = [teff, logg, fe]

    s = smsyn.smio.trilinear_interp(c,v0,v1,vi)
    
    # Broaden with rotational-macroturbulent broadening profile
    dv = restwav.loglambda_wls_to_dv(wav)

    n = 151 # Correct for VsinI upto ~100 km/s

    # Valenti and Fischer macroturb reln ERROR IN PAPER!!!
    xi = 3.98 + (teff-5770)/650
    if xi < 0: 
        xi = 0 
    
    varr,M = kernels.rotmacro(n,dv,xi,vsini)
    s = nd.convolve1d(s,M) 


    # Broaden with PSF (assume gaussian) (km/s)
    if psf > 0: s = nd.gaussian_filter(s,psf)

    spec = np.rec.fromarrays( [s,serr,wav],names='s serr w'.split() )
    return spec
