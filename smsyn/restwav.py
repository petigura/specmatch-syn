import numpy as np
from numpy import ma
from scipy.stats import nanstd, nanmean, nanmedian

from smsyn import h5plus
from smsyn import ccs
import smsyn.smio

from time import strftime
from scipy.interpolate import InterpolatedUnivariateSpline

speed_of_light = 2.99792e5 # Speed of light [km/s] 

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
    """
    Simply loads up the file from a given obs and shoves it into an h5 file.
    """

    kbcd = smsyn.smio.kbc_query(obs)
    outpath = smsyn.smio.cps_resolve(obs,'restwav')
    with h5plus.File(outpath) as h5:
        spec,header = smsyn.smio.getspec_fits(obs,type=dbtype,npad=0,header=True)
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
    d = smsyn.smio.kbc_query(obs)
    outpath = smsyn.smio.cps_resolve(obs,'restwav')

    par = coelhomatch(obs,dbtype=kwargs['dbtype'])
    mpar = dict(par.sort_values(by='ccfmax').iloc[-1])
    mpar['name'] = d['name']    
    print "%(name)s best match: teff = %(teff)i; logg = %(logg).1f" % mpar
    with h5plus.File(outpath) as h5:
        v = segdv(obs, mpar, **kwargs)
        pix = np.vstack([mid]*v.shape[0])
        h5['shift'] = np.rec.fromarrays([v,pix],names=['v','mid'])


def fitdvh5(obs, dbtype='db'):
    outpath = smsyn.smio.cps_resolve(obs,'restwav')
    with h5plus.File(outpath) as h5:
        print "saving to %s " % outpath
        shift = h5['shift'][:]
        dv = fitdv( h5['shift']['v'] ) 

        nclip = 200
        sl = slice(nclip,npix-nclip)
        specrw = h5['db'][:,sl].copy()

        fullspec = smsyn.smio.getspec_fits(obs,type=dbtype,npad=0)
        for order in range(nord):
            # Intitial guess wavelength clip off the edges, so I'm not
            # interpolating outside the limits of the spectrum

            spec = fullspec[order]
            spec = smsyn.smio.loglambda(spec)

            specrest = spec.copy()
            dlam = dv[order] / speed_of_light * specrest['w']
            specrest['w'] -= dlam # Change wavelengths to the rest wavelengths

            # Interpolate back onto conveinent grid
            specrw[order] = smsyn.smio.resamp(specrest,spec['w'][sl])

        h5['rw'] = specrw
        h5.attrs['restwav_stop_time'] = strftime("%Y-%m-%d %H:%M:%S")
        h5.attrs['restwav_sha'] = smsyn.smio.get_repo_head_sha()

def coelhomatch(obs,dbtype='db'):
    par = smsyn.smio.loadlibrary('/Users/petigura/Research/SpecMatch/library/library_coelho_restwav.csv')

    fullspec = smsyn.smio.getspec_fits(obs,type=dbtype)
    par['ccfmax'] = 0
    for i in par.index:
        p = par.ix[i]

        order = 2 # Use the MgB region to figure out which
        sl = slice(1500,3000)  # template works best

        spec = fullspec[order,sl]
        spec = smsyn.smio.loglambda(spec)
        model = smsyn.smio.getmodelseg(p,spec['w'])

        vma,xcorrma = velocityshift(model,spec)
        par.ix[i,'ccfmax'] = xcorrma

    return par

def velocityshift(model, spec, plotdiag=False):
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

    vma, xcorrma = ccs.findpeak(v,xcorr,ymax=True)

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
    
    fullspec = spec = smsyn.smio.getspec_fits(obs,type=dbtype)
    import pdb;pdb.set_trace();

    vshift = np.zeros((nord,nseg))

    # array indecies associated with different segments 
    idx_segments = np.array_split(np.arange(npix), nsegments)

    for order in range(nord):
        for iseg in range(nseg):
            

            i_segments
            sl = slice(start[iseg],stop[iseg])

            spec = fullspec[order,sl]
            spec = smsyn.smio.loglambda(spec)
            model = smsyn.smio.getmodelseg(mpar,spec['w'])

            vma, xcorrma = velocityshift(model, spec, plotdiag=plotdiag)
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


def shift_echelle_spectrum(wav, flux, ref_wav, ref_flux, nseg=8, uflux=None):
    """Shift echelle spectrum to a reference spectrum

    Given an extracted echelle spectrum having `nord` orders and
    `npix` measurements per order, shift and stretch and place on the
    wavelength scale of a reference spectrum. Shifts are determined by
    cross-correlation. In some cases, telluric lines can interfere
    with the cross-correlation peak and return outlier shift
    values. These outliers are identied and replaced with shifts
    determined from the ensemble of orders.

    Args:
        wav (array): Array with shape `(nord, npix)` with wavelengths
            of observed spectrum. Usually determined from calibration
            lamps. This does not need to be too accurate because
            spectrum will be shifted to reference spectrum wavelength
            scale.

        flux (array): Array with shape `(nord, npix)` with continuum
            normalized flux of target spectrum.

        ref_wav (array): Array with shape `(nref_wav, )` of wavelengths of 
            reference spectrum.

        ref_flux (array): Array with shape `(nref_wav, )` with the
            continuum-normalized flux of reference spectrum.

        nsegments (int): Number of segments to break each order into when 
            performing the cross-correlation.

        uflux (Optional[array]): Array with shape `(nord, npix)` with
            uncertanties of flux measurement. This array "along for
            the ride" and receives the same shift.

    Returns:
        flux_shift (array): Array with shape `(nref_wav, )` target spectrum
            shifted to the reference wavelength scale.
        uflux_shift (array): Array with shape `(nref_wav, )` of uncertainty
            array, also shifted to the reference wavelength scale
    """

    assert wav.shape==flux.shape, "wav and flux must have same shape"
    assert ref_wav.shape==ref_flux.shape, \
        "ref_wav and ref_flux must have same shape"

    nord, npix = wav.shape
    nref_wav = ref_flux.shape[0]

    vshift = np.zeros( (nord, nseg) )

    # array indecies associated with different segments 
    pix_segs = np.array_split(np.arange(npix), nseg)

    for i_order in range(nord):
        for i_seg in range(nseg):
            i_pix = pix_segs[i_seg]
            wav_seg = wav[i_order,i_pix]
            flux_seg = flux[i_order,i_pix]
            vshift[i_order,i_seg] = velocityshift2(wav_seg, flux_seg, ref_wav,ref_flux)

    return vshift


def loglambda(wav0, flux0):
    """Resample spectrum onto a constant log-lambda wavelength scale

    Args:
        wav0 (array): array of wavelengths
        flux0 (array): flux values at wav0
    
    Returns:
        wav (array): wavelengths (constant log-spacing)
        flux (array): flux values at wav
        dvel (float): spacing between measurements in velocity space (km/s)
    """
    assert wav0.shape==flux0.shape, "wav0 and flux must be same size"
    npix = wav0.size
    wav = np.logspace( np.log10( wav0[0] ), np.log10( wav0[-1] ), wav0.size)
    spline = InterpolatedUnivariateSpline(wav0, flux0)
    flux = spline(wav)
    dvel = (wav[1:] - wav[:-1])/(wav[1:]) * speed_of_light
    dvel = np.mean(dvel)
    return wav, flux, dvel

def velocityshift2(wav, flux, ref_wav, ref_flux, plot=False):
    """
    Find the velocity shift between two spectra. 

    Args:
         wav (array): Wavelength array.
         flux (array): Continuum-normalized spectrum. 
         ref_wav (array): 
         ref_flux (array): 
    
    Velocity means `spec` is red-shifted with respect to model
    """
    nwav = flux.size

    # Build spline object for resampling the model spectrum
    ref_spline = InterpolatedUnivariateSpline(ref_wav, ref_flux)

    # Convert target and model spectra to constant log-lambda scale
    wav, flux, dvel = loglambda(wav, flux)
    ref_flux = ref_spline(wav)

    flux -= np.mean(flux)
    ref_flux -= np.mean(ref_flux)

    corr = np.correlate(ref_flux, flux, mode='full')
    # Negative lag means `spec` had to be blue-shifted in order to
    # line up therefole the spectrum is redshifted with respect
    # `model`

    lag = np.arange(-nwav + 1, nwav) 
    dvel = -1*lag*dvel
    vmax, corrmax = quadratic_max(dvel, corr)

    if plot:
        from matplotlib import pylab as plt
        # Figure bookkeeping
        fig,axL = plt.subplots(ncols=2)
        gs = plt.GridSpec(1,4)

        plt.sca(axL[0])
        plt.plot(wav,ref_flux)
        plt.plot(wav,flux)

        plt.sca(axL[1])
        vrange = (-100,100)
        b = (dvel > vrange[0]) & (dvel < vrange[1])
        plt.plot(dvel[b],corr[b])
        fig.set_tight_layout(True)
        plt.draw()
        plt.show()
        plt.plot([vmax],[corrmax],'o',label='Cross-correlation Peak')

    return vmax

def quadratic_max(x, y, r=3):
    """Fit points with parabola and find peak
    
    Args:
        x (array): independent variable
        y (array): dependent variable
        r (Optional[int]): size of region to fit peak `[-r, +r]` around idxmax
    
    
    Returns:
    
    Fit a parabola near a maximum, and figure out where the derivative is 0.
    """
    idmax = np.argmax(y)
    idpeak = np.arange(idmax-r, idmax+r+1)
    a = np.polyfit(x[idpeak], y[idpeak], 2)
    xmax = -1.0 * a[1] / (2.0 * a[0]) # Where the derivative is 0
    ymax = np.polyval(a,xmax)
    return xmax, ymax
