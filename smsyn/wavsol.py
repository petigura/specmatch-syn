import numpy as np
from numpy import ma
from scipy.stats import nanstd, nanmean, nanmedian

from smsyn import h5plus
from smsyn import ccs
import smsyn.smio

from time import strftime
from scipy.interpolate import InterpolatedUnivariateSpline

SPEED_OF_LIGHT = 2.99792e5 # Speed of light [km/s] 

class VelocityShift(object):
    """Velocity Shift

    Determine pixel-by-pixel velocity shifts

    Args:
        vshift (array): Array of shape `(nord, nseg)` with velocity shifts
        pix (array): Array of shape `(nseg)` with the pixel number of the
            center of each segment
    """
    def __init__(self, nord, npix, pixmid, vshift):
        self.nord = nord
        self.npix = npix
        self.pixmid = pixmid
        self.vshift = vshift

    def caculate_dvel(self,method='global'):
        """Calculate pixel-by-pixel velocity shift

        """
        if method=='global':
            return self._calculate_dvel_global()
        
    def _calculate_dvel_global(self):
        """
        Assume that all orders have a constant dvel at a given pixel. The
        average value of the dvel for each value is used as the global
        dvel. We interpolate between the segment centers using a
        linear model.
        """

        # threshold (units of MAD) to throw out shifts 
        sigclip = 5 
        vshift = self.vshift.copy()
        med = np.median(vshift)
        mad = np.median(np.abs(vshift - med))
        bout = np.abs(vshift - med) > sigclip*mad
        vshift = ma.masked_array(vshift,bout)

        vshift = ma.mean(vshift, axis=0)
        vshift = vshift.compressed()

        if np.std(vshift) > 25:
            print "WARNING: velocity shifts not well determined"
            vshift *= 0

        pix = np.arange(self.npix)
        pfit = np.polyfit(self.pixmid,vshift,1)

        dvel = np.zeros((self.nord,self.npix)) 
        dvel += np.polyval(pfit,pix)[np.newaxis,:]
        return dvel

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
    pixmid = np.array([int(np.mean(x)) for x in pix_segs])

    print "Calculating shifts order by order"
    for i_order in range(nord):
        for i_seg in range(nseg):
            i_pix = pix_segs[i_seg]
            wav_seg = wav[i_order,i_pix]
            flux_seg = flux[i_order,i_pix]
            vshift[i_order,i_seg] = velocityshift(
                wav_seg, flux_seg, ref_wav,ref_flux
                )

    print_vshift(vshift)
    # return 
    velshift = VelocityShift(wav.shape[0], wav.shape[1], pixmid, vshift)
    dvel = velshift.caculate_dvel(method='global')

    flux_refscale = np.empty((nord, ref_wav.shape[0]))
    flux_refscale[:] = np.nan
    
    for i_order in range(nord):
        # Intitial guess wavelength clip off the edges, so I'm not
        # interpolating outside the limits of the spectrum

        # Calculate the change in wavelength to the model
        # Change wavelengths to the rest wavelengths
        dlam = dvel[i_order] / SPEED_OF_LIGHT * wav[i_order]
        wav_refscale = wav[i_order] - dlam 

        spline = InterpolatedUnivariateSpline(wav_refscale, flux[i_order])

        b = (wav_refscale[0] < ref_wav) & (ref_wav < wav_refscale[-1])
        flux_refscale[i_order,b] = spline(ref_wav[b])

    # return 
    return flux_refscale

def wav_to_dvel(wav):
    """Converts wavelengths to delta velocities using doppler formula"""
    dvel = (wav[1:] - wav[:-1]) / (wav[1:]) * SPEED_OF_LIGHT
    return dvel


def print_vshift(vshift):
    nord = vshift.shape[0]
    for i_order in range(nord):
        outstr = ["{:.2f}".format(v) for v in vshift[i_order]]
        outstr = " ".join(outstr)
        outstr = "order {:2d} ".format(i_order) + outstr
        print outstr
    

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
    dvel = (wav[1:] - wav[:-1])/(wav[1:]) * SPEED_OF_LIGHT
    dvel = np.mean(dvel)
    return wav, flux, dvel

def velocityshift(wav, flux, ref_wav, ref_flux, plot=False):
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
    dvel = -1.0 * lag*dvel
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
