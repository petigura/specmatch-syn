import os
import pandas as pd 
from astropy.io import fits
import numpy as np
import glob
from subprocess import Popen, PIPE
from scipy.io.idl import readsav
import h5py

from scipy import interpolate
import cpsutils.kbc
from smsyn.conf import kbc
#from smsyn import coelho
import smsyn.coelho
from scipy import ndimage as nd

SM_DIR = os.environ['SM_DIR']
libh5 = h5py.File("/Users/petigura/Research/SpecMatch/library/coelho_rwav.h5",'r+')
specarr = libh5['s'][:]
w = libh5['w'][:]
serr = np.ones(w.size) * 0.05


def get_repo_head_sha():
    child = Popen(
        'git --git-dir=%(SM_DIR)s.git rev-parse HEAD' % os.environ
        ,shell=True,stdout=PIPE)
    stdout, stderr = child.communicate()
    stdout = stdout.strip()
    return stdout


def kbc_query(obs):
    """
    Returns dictionary from kbc structure. Handels mc simulations gracefully
    """
    if obs.count('_')==0:
        return dict(kbc.ix[obs])
    else:
        d =  dict(kbc.ix[ obs.split('_')[0] ])
        d['obs'] = obs
        return d

def cps_resolve(obs,type):
    """
    CPS Resolve

    Parameters
    ----------
    obs : unique CPS identifier, string
    type : `restwav`, `h5`, `plotdir`,`nameobs`, `db`
    """

    # if obs is of the form rj50.373_snr=80-1 split on the _
    d = kbc_query(obs)

    d['SM_DIR'] = os.environ['SM_DIR']
    
    if type=='db':
        specpath="/Users/petigura/Research/SpecMatch/spectra/iodfitsdb/rj76.279.fits"
        return specpath

        
    if type=='restwav':
        return "%(SM_DIR)s/spectra/restwav/%(name)s_%(obs)s.h5" % d

    if type=='h5':
        try:
            d['SM_PROJDIR'] = os.environ['SM_PROJDIR']
        except KeyError:
            print "SM_PROJDIR enivornment variable not set. Using SM_DIR"
            d['SM_PROJDIR'] = os.environ['SM_DIR']
        return "/Users/petigura/Research/SpecMatch/output/h5/%(name)s_%(obs)s.h5" % d

    if type=='plotdir':
        try:
            d['SM_PROJDIR'] = os.environ['SM_PROJDIR']
        except KeyError:
            print "SM_PROJDIR enivornment variable not set. Using SM_DIR"
            d['SM_PROJDIR'] = os.environ['SM_DIR']
        return "/Users/petigura/Research/SpecMatch/output/plots/" % d

    if type=='nameobs':
        return "%(name)s_%(obs)s" % d

def getspec_fits(obs,type='db',npad=100,header=False):
    """
    Get Spectrum

    Parameters
    ----------
    obs : CPS spectrum ID
    type : What database do we read spectrum from?

    Returns
    -------
    spec : 3821 x 16 array with the spectrum
    """

    specpath = cps_resolve(obs,type)

    if type=='db':
        if header:
            header = scrape_fits_header(specpath)
        with fits.open(specpath) as hduL:
            s = hduL[0].data 
            s /= np.percentile(s,95,axis=1)[:,np.newaxis]
            serr = hduL[1].data
            w = hduL[2].data
            #w = getwav()
    nord,npix = w.shape
    pix = np.arange(npix).astype(float)
    pix = np.tile(pix,(nord,1))

    spec = np.rec.fromarrays([s,serr,w,pix],names='s serr w pix'.split())
    #spec = LE(spec) Do we need to do the byte-swapping?

    if npad!=0:
        spec = spec[:,npad:-npad]

    if header is not False:
        return spec,header
    else:
        return spec

def getspec_h5(**kw):
    """
    Get Spectrum.

    Parameters
    ----------
    type : 'cps'/'coelho'   
    ord  : order starting from 0
    wlo  : lower limit wavelength range
    whi  : upper limit

    if type is cps:
      obs  : CPS spectrum ID

    if type is coelho
      coelho
      teff
      logg
      fe
      vsini
      psf

    Returns
    -------
    spec : record array with following attributes:
           s    : Intensity
           serr : Uncertanty
           w    : Wavelength
    """

    wlo = kw['wlo']
    whi = kw['whi']
    ord = kw['ord']

    assert kw.keys().count('type')==1,"Must specify type of spectrum"

    assert kw['type']=='coelho' or kw['type']=='cps' ,\
        "type must be either cps or coelho"

    if kw['type']=='cps':
        obs = kw['obs']
        assert obs!=None,"Must specify obs"
        assert ord!=None,"Must specify order"

        if type(obs) == list and len(obs) > 1:
            obsL = obs
            allspec = []
            allwav = []
            allerr = []
            print "Stacking observations:", obsL
            for ob in obsL:
                d = dict(kbc.ix[ob])
                with h5py.File(kbc.ix[ob,'rwpath'],'r') as h5:
                    allspec.append(h5['rw'][ord,:]['s'])
                    allwav.append(h5['rw'][ord,:]['w'])
                    allerr.append(h5['rw'][ord,:]['serr'])
                    spec = h5['rw'][ord,:]
            allspec = np.vstack(allspec)
            allwav = np.mean(np.vstack(allwav), axis=0)
            allerr = np.mean(np.vstack(allerr), axis=0)

            spec['s'] = clipped_mean(allspec)
        if type(obs)== list and len(obs)==1:
            obs = obs[0]

        d = dict(kbc.ix[obs])
        with h5py.File(kbc.ix[obs,'rwpath'],'r') as h5:
            spec = h5['rw'][ord,:]
            
 
    elif kw['type']=='coelho':
 
        with h5py.File(kbc.ix[kw['targobs'],'rwpath'],'r') as h5:
            spec = h5['rw'][ord,:]
        spec = getmodelseg(kw,spec['w'])
        spec['s'] = nd.gaussian_filter1d(spec['s'],0.0)

    if wlo!=None:
        spec = spec[ (spec['w'] > wlo) & (spec['w'] < whi) ]    
    return spec


def getmodel_h5(mpar,ver=True,wrange=None):
    """
    Get model spectrum
    
    Parameters
    ----------
    mpar : dictionary with following keys:
    """
    lib = loadlibrary('/Users/petigura/Research/SpecMatch/library/library_coelho.csv')
    lib['row'] = range(len(lib))

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
        spec = smsyn.smio.getmodel(mpar,wrange=wrange)
    else:
        spec = getmodel_h5(mpar,wrange=wrange)
        
    spec = smsyn.smio.resamp(spec,w)
    return spec




def getwav(ver=False):
    wavpath = "/Users/petigura/Research/SpecMatch/config/keck_rwav.dat"
    if ver:
        print 'reading wavelength scale from %s' % wavpath
    wav = readsav(wavpath)['wav']
    return wav


def scrape_fits_header(path):
    """
    Scrape a single .fits header 
    
    Parameters
    ----------
    file : path to the fits file
    
    Returns
    -------
    d : dictionary with header values

    """
    with fits.open(path) as hduL:
        h = hduL[0].header
        d = dict(h)
        d['file'] = path
        d.pop('') # For some reason, there is an empty field
    return d

def loadlibrary(libfile):
    """
    Load SpecMatch library

    Parameters
    ----------
    libfile : path to library file (if none is given, load from
              SM_LIBRARY env variable

    

    Returns
    -------
    lib : pandas DataFrame with the following essential keys (other
          keys may be returned as well)

          - name : library name (used as an index)
          - obs  : unique CPS identifier rjXXX.XXX
          - teff : effective temperature
          - logg : surface grav.
          - fe   : [Fe/H]
          - r    : Radius [solar radii]
    """

    
    lib = pd.read_csv(libfile)
    lib.index = lib.groupby('teff logg fe'.split(),as_index=True).first().index
    return lib

def loglambda(spec0):
    """
    Resample spectrum onto a constant log-lambda wavelength scale

    Return spectrum with the same starting and ending wavelengths but
    spaced evenly in log-lambda

    Parameters
    ----------
    spec : spectrum containing w key

    """
    w0 = spec0['w']
    npix = w0.size
    spec = spec0.copy()
    
    spec['w'] = np.logspace( np.log10(w0[0]), np.log10(w0[-1]), npix)
    spec['w'][ [0,-1] ] = w0[ [0,-1] ] # 10**log10(x) sometimes does not equal x
    spec = resamp(spec0,spec['w'])
    return spec

def resamp(spec0,wnew):
    """
    Resample spectrum onto new wavelength scale
    """
    assert wnew[0] >= spec0['w'][0],"new wavelengths outside of old"
    assert wnew[-1] <= spec0['w'][-1],"new wavelengths outside of old"


    names = spec0.dtype.names
    arrL = []
    for n in names:
        arrL+=[np.zeros(wnew.size)]

    spec = np.rec.fromarrays(arrL,names=names)
    spec['w'] = wnew
    for k in ['s','serr','pix']:
        if names.count(k)==1:
            spec[k] = spline(spec['w'],spec0['w'],spec0[k])

    return spec

def spline(xi,xp,fp):
    """
    Convenience standin for dspline
    """
    # (Cubic) spline back onto original wavelength scale.
    tck     = interpolate.splrep(xp,fp)
    return  interpolate.splev(xi,tck)



def getmodelseg(mpar,w):
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
    wrange[0]-=1
    wrange[-1]+=1

    spec = getmodel(mpar,wrange=wrange)
    spec = resamp(spec,w)
    return spec

def getmodel(mpar, ver=True, wrange=None):
    """
    Get model spectrum
    
    Parameters
    ----------
    mpar : dictionary with following keys:
           - type
           - path
    """
    
    # Get model path
    model = mpar['type']
    path = "/Users/petigura/Research/SpecMatch/" + mpar['path']
    
    fL = glob.glob(path)

    if model=='coelho':
        dw = 0.01999596
        w0 = 2990.015161
        
        s = fits.open(path)[0].data[0]
        w = np.arange(s.size) * dw + w0
    elif model=='pheonix':
        s = fits.open(path)[0].data
        path = 'pheonix/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits'

        w = fits.open(path)[0].data
        w = vac2air(w)
    elif model=='bt':
        df = pd.read_fwf(path,colspecs=[(2,23),(24,35)],names=['w','s'])
        for k in df.columns:
            df[k] = df[k].str.replace('D','e')
            df[k] = df[k].convert_objects(convert_numeric=True)

        df = df.sort('w')
        w = np.array(df.w)
        w = vac2air(w)
        s = np.array(df.s)
        s = 10**s

    serr = np.ones(w.size) * 0.05

    spec = np.rec.fromarrays([s,serr,w],names=['s','serr','w'])
    if wrange is None:
        return spec

    brange = (spec['w'] > wrange[0]) & (spec['w'] < wrange[1])
    spec = spec[brange]
    return spec


