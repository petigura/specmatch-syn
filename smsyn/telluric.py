import copy 

import numpy as np
import pandas as pd
import lmfit
from matplotlib.gridspec import GridSpec
from matplotlib.pylab import plt

from smsyn import smio
from smsyn import continuum
from smsyn import pdplus



linelistpath = '/Users/petigura/Research/SpecMatch/config/telluric_w+d.csv' 
tell = pd.read_csv(linelistpath,index_col=0)
nlines = len(tell)

def telluric_psf(obs,plot=False,plot_med=False):
    """
    Telluric PSF
    
    Measure the instrumental profile of HIRES by treating telluric
    lines as delta functions. The width of the telluric lines is a
    measure of the PSF width.

    Method
    ------ 
    Fit a comb of gaussians to the O2 bandhead from 6270-6305A. The
    comb of gaussians that describes the telluric lines has 4 free
    parameters:

       - sig  : width of the line
       - wls0 : shift of telluric line centers: -0.1 to +0.1 angstroms
       - wls1 : wavelength stretch allows for 1% fluctuation in dlam/dpix
              0.99 to 1.01.
       - d    : scale factor to apply to the telluric lines

    If wls0 is off by more than a line width, we might not find a
    solution. We perform a scan in wls0.
    
    Parameters
    ----------
    obs : CPS observation id.
    """
    
    # Load spectral region and prep.
    spec = smio.getspec(obs=obs)[14]
    spec = spec[(spec['w'] > 6270) & (spec['w'] < 6305)]
    spec['s'] /= continuum.cfit(spec)

    # figure out where the 95 percentile is for data that's this
    # noisey, then adjust continuum
    serr = np.median(spec['serr'])
    contlevel = np.percentile(np.random.randn(spec.size)*serr,95)
    spec['s'] *= (1+contlevel)
    darr = np.array(tell.d)

    # Intialize parameters
    p0 = lmfit.Parameters()
    p0.add('wls0',value=0.0)
    p0.add('sig',value=0.02,min=0,max=0.1)
    p0.add('wls1',value=1.0,min=0.99,max=1.01)
    p0.add('d',value=1,min=0.25,max=2)

    def model(p):
        return telluric_comb(p['sig'].value,p['wls0'].value,p['wls1'].value,
                             spec['w'],tell.wcen,darr=p['d'].value*darr)

    def res(p):
        """
        Residuals

        Returns residuals for use in the LM fitter. I compute the
        median absolute deviation to flag outliers. I remove these
        values from the residual array. However, I divide the
        residuals by the total number of residuals (post sigma
        clipping) so a to not penalize solutions with more points.

        Parameters
        ----------
        p : lmfit parameters object

        Returns
        -------
        res : Residual array (used in lmfit)
        
        """
        mod = model(p)
        res = (spec['s'] - mod) / spec['serr'] # normalized residuals

        mad = np.median(np.abs(res))
        b =  (np.abs(res) < 5*mad ) 
        res /= b.sum()

        if b.sum() < 10:
            return res * 10
        else:
            return res[b]

    # Do a scan over different wavelength shifts
    wstep = 0.01
    wrange = 0.1
    wls0L = np.arange(-wrange,wrange+wstep,wstep)

    chiL = np.zeros(len(wls0L))
    outL = []
    for i in range(len(wls0L)):
        p = copy.deepcopy(p0)
        p['wls0'].value = wls0L[i]
        out = lmfit.minimize(res,p)
        chiL[i] = np.sum(res(p)**2)
        outL+=[out] 

    # If the initial shift is off by a lot, the line depths will go to
    # zero and median residual will be 0. Reject these values
    out = outL[np.nanargmin(chiL)]
    p = out.params
    lmfit.report_fit(out.params)

    # Bin up the regions surronding the telluric lines
    wtellmid = np.mean(tell.wcen)
    def getseg(wcen):
        wcen = wtellmid + (wcen-wtellmid)*p['wls1'].value + p['wls0'].value
        dw = spec['w'] - wcen
        b = np.abs(dw) < 0.3

        seg = pd.DataFrame({'dw':dw,'s':spec['s'],'model':model(p)})
        seg = pdplus.LittleEndian(seg.to_records(index=False))
        seg = pd.DataFrame(seg)
        return seg[b]
    
    seg = map(getseg,list(tell.wcen))
    seg = pd.concat(seg,ignore_index=True)
    wstep = 0.025
    bins = np.arange(-0.3,0.3+wstep,wstep)
    seg['dw0'] = 0.
    for dw0 in np.arange(-0.3,0.3,wstep):
        seg['dw0'][seg.dw.between(dw0,dw0+wstep)] = dw0
    bseg = seg.groupby('dw0',as_index=False).median()

    mod = model(p) # best fit model
    def plot_spectrum():
        # Plot telluric lines and fits
        plt.plot(spec['w'],spec['s'],label='Stellar Spectrum')
        plt.plot(spec['w'],mod,label='Telluric Model')
        plt.plot(spec['w'],spec['s']-mod+0.5,label='Residuals')
        plt.xlim(6275,6305)
        plt.ylim(0.0,1.05)
        plt.xlabel('Wavelength (A)')
        plt.ylabel('Intensity')
        plt.title('Telluric Lines')
        
    def plot_median_profile():
        # Plot median line profile
        plt.plot(bseg.dw,bseg.s,'.',label='Median Stellar Spectrum')
        plt.plot(bseg.dw,bseg.model,'-',label='Median Telluric Model')
        plt.title('Median Line Profile')
        plt.xlabel('Distance From Line Center (A)')
        yl = list(plt.ylim())
        yl[1] = 1.05
        plt.ylim(*yl)

    if plot:
        # Used for stand-alone telluric diagnostic plots
        gs = GridSpec(1,4)
        fig = plt.figure(figsize=(12,3))
        plt.gcf().set_tight_layout(True)
        plt.sca(fig.add_subplot(gs[0,:3]))
        plot_spectrum()
        plt.sca(fig.add_subplot(gs[0,3]))
        plot_median_profile()

    if plot_med:
        # Used in quicklook plot
        plot_median_profile()


    # sig = width of telluric line in
    sig = p['sig'].value # [Angstroms]
    sig = sig/6290*3e5 # [km/s]
    sig = sig/1.3 # [pixels]
    return sig

def telluric_comb(sig,wls0,wls1,w,wcen,s=None,darr=None):
    """
    Synthesize a model spectrum given a telluric line list

    sig  : gaussian sigma for all the telluric lines
    wls0 : wavelength shift
    wls1 : wavelength stretch
    w    : Evaluate the comb at these wavelengths
    darr : depth of each telluric line

    """
    wtellmid = np.mean(wcen)
    wcenL = wtellmid + (wcen-wtellmid)*wls1 + wls0
    wcenL = list(wcenL)
    
    model = [ -1.0*gaussian(w,1,wcen,sig) for wcen in wcenL ]
    model = np.vstack(model)
    if darr is None:
        darr,res,rank,s = np.linalg.lstsq(model.T,s-1)
        model = np.dot(model.T,darr)
        model += 1
        return model,darr
    else:
        model = np.dot(model.T,darr)
        model += 1
        return model

def gaussian(x, A, x0, sig):
    val = A * np.exp(-(x - x0)**2 / 2. / sig**2)
    return val
