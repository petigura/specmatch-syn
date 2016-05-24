import os
from smsyn import smio
import h5py
import numpy as np

SM_DIR = os.environ['SM_DIR']
lib = smio.loadlibrary('/Users/petigura/Research/SpecMatch/library/library_coelho.csv')

#lib = library.loadlibrary('library/library_coelho_4900-9000.csv')
lib['row'] = range(len(lib))

# h5 directory with all the library spectra

libh5 = h5py.File("/Users/petigura/Research/SpecMatch/library/coelho_rwav.h5",'r+')
#libh5 = h5py.File(os.environ['SM_DIR']+'library/coelho_4900-9000.h5','r+')
specarr =libh5['s'][:]
w = libh5['w'][:]
serr = np.ones(w.size) * 0.05


def getmodel(mpar,ver=True,wrange=None):
    """
    Get model spectrum
    
    Parameters
    ----------
    mpar : dictionary with following keys:
    """
    
    idx = (mpar['teff'],mpar['logg'],mpar['fe'])
    s = specarr[ lib.ix[idx,'row'] ]
    
    spec = np.rec.fromarrays([s,serr,w],names=['s','serr','w'])

    if wrange is None:
        return spec

    brange = (spec['w'] > wrange[0]) & (spec['w'] < wrange[1])
    spec = spec[brange]
    return spec

def getmodelseg(mpar,w, libfull=False):
    """
    Get Model Segment
    
    Thin wrapper around getmodel. After model is loaded up,
    interpolate onto `w` wavelength scale

    Parameters
    ----------
    mpar : model parameters, passed to getmodel
    w : wavelength array

    Returns
    -------
    spec : spectrum in record array form interpolated on to give wavelength 
           scale
    """

    # Select a region just a little bigger than the final segment
    wrange = w[[0,-1]]
    wrange[0]-=2
    wrange[-1]+=2

    if libfull:
        spec = smio.getmodel(mpar,wrange=wrange)
    else:
        spec = getmodel(mpar,wrange=wrange)
        
    spec = smio.resamp(spec,w)
    return spec
