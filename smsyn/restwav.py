import numpy as np
from numpy import ma
from scipy.stats import nanstd, nanmean, nanmedian

from smsyn import smio
from smsyn import h5plus
from smsyn import ccs

from time import strftime

speed_of_light = 2.99792e5 # Speed of light [km/s] 

nord = 16
nseg = 7
segsize = 1000
segstep = 500

mid = np.arange(nseg)*segstep + segsize/2.
start = np.arange(nseg)*segstep
stop = start+segsize
npix = 4021

def restwav(obs):
    if obs.startswith('ra'): dbtype = 'apf'
    else: dbtype = 'db'
        
    print "Converting to HD5"
    obs2h5(obs, dbtype=dbtype)
    print "Finding velocity shifts"
    segdvh5(obs, dbtype=dbtype)
    print "Fitting velocity shifts"
    fitdvh5(obs, dbtype=dbtype)
    
def obs2h5(obs, dbtype='db'):
    kbcd = smio.kbc_query(obs)
    outpath = smio.cps_resolve(obs,'restwav')
    with h5plus.File(outpath) as h5:
        spec,header = smio.getspec_fits(obs,type=dbtype,npad=0,header=True)
        h5['db'] = spec
        header = dict(header,**kbcd)
        h5plus.dict_to_attrs(h5,header)
def segdvh5(obs,**kwargs):
    """
    Segment Velocity Shift HDF5 wrapper

    Determines the closest matching model using coelho match. Then
    runs segdv and saves the shifts in a h5 directory.
    
    Parameters
    ----------
    obs : CPS identifier
    kwargs : keyword arguments are passed to segdv
    """
    d = smio.kbc_query(obs)
    outpath = smio.cps_resolve(obs,'restwav')

    par = coelhomatch(obs,dbtype=kwargs['dbtype'])
    mpar = dict(par.sort_values(by='ccfmax').iloc[-1])
    mpar['name'] = d['name']    
    print "%(name)s best match: teff = %(teff)i; logg = %(logg).1f" % mpar
    with h5plus.File(outpath) as h5:
        v = segdv(obs,mpar,**kwargs)
        pix = np.vstack([mid]*v.shape[0])
        h5['shift'] = np.rec.fromarrays([v,pix],names=['v','mid'])


def fitdvh5(obs, dbtype='db'):
    outpath = smio.cps_resolve(obs,'restwav')
    with h5plus.File(outpath) as h5:
        print "saving to %s " % outpath
        shift = h5['shift'][:]
        dv = fitdv( h5['shift']['v'] ) 

        nclip=200
        sl = slice(nclip,npix-nclip)
        specrw = h5['db'][:,sl].copy()

        fullspec = smio.getspec_fits(obs,type=dbtype,npad=0)
        for order in range(nord):
            # Intitial guess wavelength clip off the edges, so I'm not
            # interpolating outside the limits of the spectrum

            spec = fullspec[order]
            spec = smio.loglambda(spec)

            specrest = spec.copy()
            dlam = dv[order] / speed_of_light * specrest['w']
            specrest['w'] -= dlam # Change wavelengths to the rest wavelengths

            # Interpolate back onto conveinent grid
            specrw[order] = smio.resamp(specrest,spec['w'][sl])

        h5['rw'] = specrw
        h5.attrs['restwav_stop_time'] = strftime("%Y-%m-%d %H:%M:%S")
        h5.attrs['restwav_sha'] = smio.get_repo_head_sha()



def coelhomatch(obs,dbtype='db'):
    par = smio.loadlibrary('/Users/petigura/Research/SpecMatch/library/library_coelho_restwav.csv')

    fullspec = smio.getspec_fits(obs,type=dbtype)
    par['ccfmax'] = 0
    for i in par.index:
        p = par.ix[i]

        order = 2 # Use the MgB region to figure out which
        sl = slice(1500,3000)  # template works best

        spec = fullspec[order,sl]
        spec = smio.loglambda(spec)
        model = smio.getmodelseg(p,spec['w'])

        vma,xcorrma = velocityshift(model,spec)
        par.ix[i,'ccfmax'] = xcorrma

    return par




def velocityshift(model,spec,plotdiag=False):
    """
    Find the velocity shift between two spectra. 
    
    Velocity means `spec` is red-shifted with respect to model
    """

    dv = loglambda_wls_to_dv(spec['w'])
    assert (model['w']==spec['w']).all(),"wavelength arrays must be same"

    dv = (spec['w'][1:] - spec['w'][:-1])/(spec['w'][1:]) * speed_of_light
    mdv = np.mean(dv)
    assert dv.ptp() / mdv < 1e-6,"wavelengths must be log-lambda"
    dv = mdv

    modelms = model['s'] - np.mean(model['s']) # Model, mean subtracted.
    specms = spec['s'] - np.mean(spec['s']) # Model, mean subtracted.
    npix = model['w'].size

    xcorr = np.correlate(modelms,specms,mode='full')
    # Negative lag means `spec` had to be blue-shifted in order to
    # line up therefole the spectrum is redshifted with respect
    # `model`

    lag = np.arange(-npix+1,npix) 
    v = -1*lag*dv

    vma,xcorrma = ccs.findpeak(v,xcorr,ymax=True)

    if plotdiag:
        # Figure bookkeeping
        fig = gcf()
        clf()
        gs = GridSpec(1,10)

        axL = [fig.add_subplot(g) for g in [gs[0:8], gs[-1]]]
#        axL = [fig.add_subplot(211),fig.add_subplot(212)]
        gcf().set_tight_layout(True)

        sca(axL[0])
        plot(model['w'],model['s'])
        plot(spec['w'],spec['s'])

        sca(axL[1])
        vrange = (-100,100)
        b = (v > vrange[0]) & (v < vrange[1])
        plot(v[b],xcorr[b])

        savefig('dv_%f.png' % mdv)
        draw()
        show()
        
    return vma,xcorrma



def loglambda_wls_to_dv(w, nocheck=True):
    """
    Checks that spectrum is on a constant loglambda wavelength
    scale. Then, returns dv
    """
    dv = (w[1:] - w[:-1])/(w[1:]) * speed_of_light
    mdv = np.mean(dv)
    if not nocheck: assert dv.ptp() / mdv < 1e-6,"wavelengths must be log-lambda"
    dv = mdv
    return dv


def segdv(obs,mpar,plotdiag=False,dbtype='db'):
    """
    Segment Velocity Shift

    For each order, compute the velocity shift between the coelho
    models and HIRES spectra.

    Parameters
    ----------
    obs : CPS identifier
    mpar : model parameters (passed to getmodelseg)
    diagplot : diagnostic plot

    Returns 
    -------
    vshift : nord x nseg array with the velocity shifts for each segment.
    """
    
    fullspec = spec = smio.getspec_fits(obs,type=dbtype)
    vshift = np.zeros((nord,nseg))
    for order in range(nord):
        for iseg in range(nseg):
            sl = slice(start[iseg],stop[iseg])

            spec = fullspec[order,sl]
            spec = smio.loglambda(spec)
            model = smio.getmodelseg(mpar,spec['w'])

            vma,xcorrma = velocityshift(model,spec,plotdiag=plotdiag)
            print order,start[iseg], vma
            vshift[order,iseg] = vma

    return vshift

def fitdv(vshift):
    """
    Fit velocity shift
    
    Mask out dv's that are large outliers with respect to the rest of
    the segments. The overall WLS is determed by averaging the shift
    amounts across diffent orders.

    Parameters
    ----------
    vshift : nord x nseg array with shifts for each segment,

    Returns
    -------
    dv : shift in velocity for all pixels.

    """

    sigclip = 5 # threshold (units of MAD) to throw out shifts 
    med = nanmedian(vshift)
    mad = nanmedian(np.abs(vshift -med))
    bout = np.abs(vshift - med) > sigclip*mad

    vshift = ma.masked_array(vshift,bout)
    dv = np.zeros((nord,npix))
    pix = np.arange(npix)
    
    vshifto = ma.mean(vshift, axis=0)
    x = mid[~vshifto.mask]
    y = vshifto.compressed()

    if np.std(y) > 25:
        print "WARNING: velocity shifts not well determined"
        y *= 0
    
    np.set_printoptions(precision=2)
    print "mid pix",x
    print "dv km/s",y

    pfit = np.polyfit(x,y,1)
    dv += np.polyval(pfit,pix)
    return dv

