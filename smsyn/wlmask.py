import pandas as pd 
import numpy as np
from numpy import ma
from scipy import ndimage as nd
import os

def loadmask(maskpath='%s/config/wav_mask.csv' % os.environ['SM_DIR']):
    """
    Load Mask

    Parameters
    ----------

    maskpath : csv file that contains wavelength mask. Mus be of the
               following form:

               start,stop
               # Order 4890
               5018, 5019.5   # Wings of lines looked bad
               5027.5, 5028.5 # Wings of lines looked bad
    
               Can comment out with #

    Returns
    -------
    dfmask : DataFrame with the start and stop of masked regions.
    """

    dfmask = pd.read_csv(maskpath, comment='#')
    dfmask = dfmask.dropna()
    return dfmask

def getmask(w,dfmask,mode='exclude'):
    """
    Get Masked Array
    
    Parameters
    ----------
    w : range of wavelengths
    dfmask : DataFrame with the following parameters
             - start
             - stop
    mode : 'include' / 'exclude'

    Return
    ------
    mask : True exclude element

    """

    assert (mode=='exclude') | (mode=='include'),'must select include|exclude'

    mask = np.zeros(w.size).astype(bool)
    for i in  dfmask.index:
        d = dfmask.ix[i]
        mask = mask |  (w > d['start']) & (w < d['stop'])

    if mode=='include':
        mask = ~mask

    return mask

def specmask(spec,mspec=None):
    """
    Mask spectrum
    
    Assume spectrum is continuum normalized. Search for airglow by
    looking for regions with intensity higher than 1.2.
    
    Parameters 
    ----------
    spec : normalized spectrum
    """
    s = spec['s'] 
    w = spec['w'] 
    mask = s > 1.2

    if mspec is not None:
        res = s-mspec
#        res -= nd.median_filter(res,5) # filter residuals
        pres = np.abs(res)
        mad = np.median(pres)
        mask = mask | (pres > mad*10)

    # Grow mask by 10 pixels in both dimensions
    mask = np.convolve(mask,np.ones(20),mode='same') > 0
    
    # Identifies the contiguous masked regions
    sL = ma.notmasked_contiguous(ma.masked_array(s,~mask))

    if sL is not None:
        data = [ (w[slice.start],w[slice.stop-1])  for slice in sL]
        dfmask = pd.DataFrame(data=data,columns=['start','stop'])
        return dfmask

                              
        
