import os
import sys 
from time import strftime
from cStringIO import StringIO as sio

import numpy as np
from scipy import optimize
import scipy.ndimage as nd
import pandas as pd 
import h5py
from multiprocessing import Pool
from matplotlib import mlab

from smsyn import smio
from smsyn import coelho
from smsyn import rotbro
from smsyn import fftspecfilt
from smsyn import h5plus
from smsyn import pdplus
import cpsutils.kbc

SM_DIR = os.environ['SM_DIR']
kbc = cpsutils.kbc.loadkbc()
kbc['h5bname'] = kbc.name+"_"+kbc.obs+".h5"
kbc['rwpath'] = os.path.join(SM_DIR,"spectra/restwav/")
kbc['rwpath'] = kbc['rwpath']+ kbc['h5bname']

def get_segdf():
    """
    Get Segment DataFrame
    
    Returns
    -------

    segdf : DataFrame with the wavelength regions [wlo,whi] and orders
            for the wavelength segments under consideration for SpecMatch.
    """

    s = """\
wlo   whi   ord
4980  5060  0
5055  5130  1
5125  5205  2
5160  5190  2
5200  5280  3
5280  5360  4
5360  5440  5
5445  5520  6
5530  5610  7
5620  5700  8
5710  5790  9
5800  5880  10
5900  5980  11
6000  6080  12
6100  6190  13
6150  6170  13
6159  6165  13
6210  6260  14
6320  6420  15"""

    segdf = pd.read_table(sio(s),sep=' ',comment='#', skipinitialspace=True)
    segdf = segdf.dropna()
    for k in 'wlo whi ord'.split():
        segdf[k] = segdf[k].astype(int)
        segdf.index = segdf.wlo
    return segdf

def coelho_match(obs,snr=None,h5path=None,verbose=True,debug=False,numpro=1):
    """
    Coelho SpecMatch

    Compare target spectrum to library spectra. Compute chi2 for each
    comparison. Comparison is done in two steps:
    
    1. Compare with solar metalicity model spectra and let VsinI float
       as a free parameter. VsinI captures other broadening terms too.
    2. Compare with all model spectra, adopting VsinI from Step 1.

    Parameters
    ----------
    obs     : input observation name - string
    lib     : SpecMatch library

    History
    -------
    Jan 20 2014 : Written by Erik Petigura
    Mar 10 2014 : EAP - broke up into several helper functions

    """
    print "Running SpecMatch on %(name)s %(obs)s" % smio.kbc_query(obs)
    
    lib = smio.loadlibrary('/Users/petigura/Research/SpecMatch/library/library_coelho.csv')
    segdf = get_segdf()
    
    # For debugging
    # Target 20 chi2 comparisons, minimum required to for code to proceed
    if debug:
        lib = lib.iloc[::40]

        print """
##### Debugging ######"
Running with a paired down library with %i model spectra
""" % len(lib)

    
    wloL = [5200, 5360, 5530, 6100, 6210]
    segdf = segdf.ix[wloL]

    smpar = lib_seg_cross(lib,segdf)
    smpar['targobs'] = obs

    if snr!=None:
        print "Adding noise to simulate SNR=%i spectrum" % snr
        
    smpar['vsini'] = None

    if numpro==1:
        smres,arrL = matchloop(smpar)
    else:
        # Split the SM input parameter list into a bunch of small
        # chuncks. Some of the wavelength sections take longer than
        # others to process. Making the chunck size smaller will keep
        # each core busy.
        idxL = np.array_split(smpar.index,numpro*4)
        smparL = [smpar.ix[idx] for idx in idxL]

        # Using Pool.map to handle parallelism. 
        pool = Pool(numpro)
        resL = pool.map(matchloop,smparL)
        pool.close()
        pool.join()

        smres = pd.concat([r[0] for r in resL])
        arrL = reduce(lambda x,y : x+y, [r[1] for r in resL] )

    if h5path is not None:
        h5store(h5path,smres,arrL)
        # copy over attributes
        h5path0 = smio.cps_resolve(obs,'restwav')
        h5plus.copy_attrs(h5path0,h5path)

    return smres,arrL


def lib_seg_cross(lib,segdf):
    """
    Library/Segment Cross Product

    To compute chi for all the spectra in library and all the
    segments, we need to compute the cartestian cross-product of lib
    and segdf.

    """
    # smpar is a DataFrame specifying the parameters for each of the matches
    smpar = lib.copy()
    smpar['libidx'] = range(len(smpar))
    smpar = smpar.rename(columns={'name':'libname'})
    segdf['key'] = 1
    smpar['key'] = 1
    smpar = pd.merge(smpar,segdf).sort_values(by=['wlo','libidx'])
    smpar.index = np.arange(len(smpar))
    return smpar

def matchloop(smpar,verbose=True,snr=None):
    """
    Match Loop
    
    Loops over the elements in the smpar DataFrame. Also performs some
    counting and display

    Parameters
    ----------
    smpar : DataFrame with the following keys
            - type
            - wlo
            - whi
            - ord
            - teff  (library parameters)
            - logg
            - fe 
            - targobs

    Returns
    -------
    smres : DataFrame (same length as smpar) with results (like figure
            of merit) returned at each iteration.
    arrL : List of 1D array output from getfom. These can be different
           lengths for different segments, so no use turning them 2D.
    """
    counter = 0 
    nmatch = len(smpar)

    scldL = [] # Stores the scalar results
    arrL = [] # Stores the array intermediate steps (option to save later)
    for i in smpar.index:
        smpard = dict(smpar.ix[i])

        # Load up target and library spectrum
        kw = smpard
        libspec = getspec(**kw)

        kw['type'] = 'cps'
        targspec = getspec(obs=smpard['targobs'], **kw)

        if snr!=None:
            np.random.seed(0)
            targspec['s'] += np.random.randn(*targspec['s'].shape)/snr
            targspec['serr'] = 1./snr

        scld,arr = getfom(targspec,libspec,vsini=smpard['vsini'])
        scldL += [scld]
        arrL += [arr]

        counter += 1
        s =  "%03i/%03i " % (counter,nmatch) + \
          " %(libname)10s %(teff)i %(logg).2f %(fe)+.2f " % smpard +\
          " %(fchi)5.1f \n " % scld
              
        if verbose:
            sys.stderr.write(s)

    smres = pd.DataFrame(scldL,index=smpar.index)
    smres = pd.concat([smpar.drop('vsini',axis=1),smres],axis=1)
    return smres,arrL

def getspec(**kw):
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
        spec = coelho.getmodelseg(kw,spec['w'])
        spec['s'] = nd.gaussian_filter1d(spec['s'],0.0)

    if wlo!=None:
        spec = spec[ (spec['w'] > wlo) & (spec['w'] < whi) ]    
    return spec


def getfom(tspec,lspec,vsini=None):
    """
    Get figure of merit.

    Parameters
    ----------
    tspec : target spectrum
    lspec : library spectrum 
    vsini : before computing match (spin up library spectrum by
            vsini). If left as None, we let vsini float as a free
            parameter

    Returns
    -------
    sclrd : dictionary of scalar output parameters
    arr : record array with array output parameters
    """

    err = np.sqrt(tspec['serr']**2 + lspec['serr']**2)
    npix = tspec.size
        
    def model(vsini):
        lspecb = lspec.copy() # Broadened library spectrum
        lspecb['s'] = rotbro.rotbro(lspec['w'],lspec['s'],np.mean(lspec['w']),
                                    vsini)
        return lspecb

    def obj(parL):       
        lspecb = model(parL[0])
        res = tspec['s'] - lspecb['s']
        fres = fftspecfilt.fftbandfilt(res,whi=400)
        fchi = np.sum((fres/err)**2) / npix
        return fchi

    if vsini==None:
        pbest = optimize.fmin(obj,[5],disp=0)
        vsini = np.abs(pbest[0])
    #print vsini
        
    lspecb = model(vsini)
    res = tspec['s']-lspecb['s']
    fres = fftspecfilt.fftbandfilt(res,whi=400)
    fchi = np.sum((fres/err)**2) / npix

    # Pack scalar output into a dictionary, array output into a record array
    sclrd = dict(fchi=fchi,vsini=vsini)
    arr = dict(res=res,fres=fres,tspec=tspec['s'],lspec=lspecb['s'],
               w=tspec['w'])

    arr = np.array(pd.DataFrame(arr).to_records(index=False))        
    return sclrd,arr


def h5store(h5path,smres,arrL,ntop=20):
    """
    Store outputs from SpecMatch into an h5 file.


    Parameters
    ----------
    h5path : where to store the results?
    smres : DataFrame with the scalar outputs (like chi) for each match
    arrL : array outputs for each match
    """

    outdict = splitsegs(smres,arrL,narr=20)         
    with h5plus.File(h5path,'c') as h5:
        for wlo in outdict.keys():
            h5.create_group(wlo)

            # Arrays corresponding to given wavelength region
            garr = outdict[wlo]['arr'] 

            # Split them up according to library vs. target spectrum
            # - lspec : ntop x npix record array with library spectra 
            # - tspec : npix record array with template spectrum
            lspec = mlab.rec_drop_fields(garr,['w','tspec'])
            tspec = mlab.rec_keep_fields(garr,['w','tspec'])[0]

            h5.create_dataset('%s/lspec' % wlo,compression=1,
                              data=lspec,shuffle=True)
            h5.create_dataset('%s/tspec' % wlo,compression=1,
                              data=tspec,shuffle=True)

            gsmres = outdict[wlo]['smres']
            gsmres = pdplus.df_to_ndarray(gsmres)

            h5.create_dataset('%s/smres' % wlo,data=gsmres)
        h5.attrs['specmatch_stop_time'] = strftime("%Y-%m-%d %H:%M:%S")
        h5.attrs['specmatch_sha'] = smio.get_repo_head_sha()


def splitsegs(smres,arrL,narr=20):
    """
    Split segments

    Since arrays can be different lengths for different wavelength
    regions, we split smres and arrL according to their wavelength
    region. Return a dict of dicts

    Parameters
    ----------
    smres : DataFrame with the scalar outputs (like chi) for each match
    arrL : array outputs for each match
    

    Returns
    -------
    outdict : dictionary with different wavelength regions keys i.e.
              {'5200': dict(smres=smres,arrL=arrL),
                ...
               '6100': dict(smres=smres,arrL=arrL)}
    """

    # Store the different wavelength segments seperately
    g = smres.groupby('wlo')
    outdict = {}
    for k in g.groups.keys():
        gname = str(k)
        ig = g.groups[k] # indecies in current group
        gsmres = smres.ix[ig]
        gsmres = gsmres.sort_values(by='fchi')

        if (len(ig) < narr):
            narr=len(ig)

        # Group arrays
        bestidx = gsmres.index[:narr] # indecies of the best matches
        garr = [ arrL[i] for i in bestidx ] 
        garr = np.vstack(garr)

        # Explicity denote which spectra I've written out
        gsmres['arrLrow'] = -1 
        gsmres.ix[bestidx,'arrLrow'] = np.arange(narr)

        gdict = dict(smres=gsmres,arr=garr)
        outdict[gname] = gdict

    return outdict
