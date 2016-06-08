#!/usr/bin/env python

import numpy as np
from scipy.stats import nanmean, nanmedian
import os
import itertools
from astropy.io import fits

import smsyn

def read_fits(specfile, flatten=True, Bflat=False, nflat=False,geterr=False,order=6,specorders=None, verbose=False):
    """Read APf fits spectrum
    
    Read an APF spectrum from the raw, unnormalized fits file
    and return a normalized 2d spectrum and uncertainties. Use continuum
    derirved from B star observations to compensate for the blaze function.

    Args:
        specfile (string): name of input fits file
        flatten (bool): (optional) Deblaze?
        Bflat (bool): (optional) Use B star observations for continuum normalization
        nflat (bool): (optional) Use narrow flat for continuum normalization
        geterr (bool): (optional) Also return uncertainties?
        order (int): (optional) Polynomial order for continuum normalization, used only if Bflat is False
        specorders (list): (optional) List of ints to only load specified spectral orders
        verbose (bool): (optional) Print extra messages

    Returns:
        array: N orders x N pixel array with the spectrum

    """
        
    spec = fits.getdata(specfile)


    if specorders is None:
        specorders = range(spec.shape[0])

    spec = spec[specorders,:]
    if geterr:
        errspec = np.sqrt(spec)
        if flatten and not nflat:
            errspec /= spec

    if flatten and not Bflat and not nflat:
        if verbose: print "Polynomial continuum normalization. order = %d" % order
        for i in range(spec.shape[0]):
            spec[i] = flatspec(spec[i],order=order)
    elif Bflat:
        if verbose: print "B star continuum normalization"

        spec = spec[:,:-1]
        errspec = errspec[:,:-1]
        
        fitsfile = os.path.join(os.path.dirname(os.path.abspath(smsyn.__file__)),'inst/apf/Bcont.fits')
        bspec = read_fits(fitsfile, flatten=False, Bflat=False, specorders=specorders)

        for i in range(spec.shape[0]):
            spec[i] = Bflatten(spec[i], bspec[i])
    elif nflat:
        if verbose: print "Narrow flat continuum normalization"
        fitsfile = os.path.join(os.path.dirname(os.path.abspath(smsyn.__file__)),'inst/apf/Nflat.fits')
        bspec = read_fits(fitsfile, flatten=False, Bflat=False, specorders=specorders)
        
        for i in range(spec.shape[0]):
            if geterr: errspec[i] = np.sqrt(spec[i]) / spec[i]
            spec[i] = Bflatten(spec[i], bspec[i])
            
    if geterr: return spec, errspec
    else: return spec

def flatspec(rawspec, order=4):

    ffrac = 0.98

    x = np.arange(0,len(rawspec),1)

    coef = np.polyfit(x,rawspec,order)
    poly = np.poly1d(coef)
    yfit = poly(x)

    for i in range(8):
        normspec = rawspec / yfit
        pos = np.where((normspec >= ffrac) & (normspec <= 1.2))[0]
        coef = np.polyfit(x[pos],rawspec[pos],i+2)
        poly = np.poly1d(coef)
        yfit = poly(x)
        
    #pl.plot(rawspec, 'k-')
    #pl.plot(x,yfit,'b-')
    #pl.show()

    normspec = rawspec / yfit

    return normspec

def Bflatten(rawspec, bspec, order=12):
    ffrac = 0.98

    x = np.arange(0,len(bspec),1)

    bspec /= nanmean(bspec)
    rawspec /= nanmean(rawspec)


    normspec = rawspec / bspec
    
    cont = normspec[normspec >= 0.8]
    normspec /= nanmedian(cont)
    
    #normspec /= np.max(sorted(normspec[pos])[:-3])

    return normspec

