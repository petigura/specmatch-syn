import numpy as np
from numpy import ma
from scipy.stats import nanstd, nanmean, nanmedian
from scipy.interpolate import InterpolatedUnivariateSpline

SPEED_OF_LIGHT = 2.99792e5 # Speed of light [km/s] 

def velocityshift(wav, flux, ref_wav, ref_flux, plot=False):
    """
    Find the velocity shift between two spectra. 

    Args:
        wav (array): Wavelength array.
        flux (array): Continuum-normalized spectrum. 
        ref_wav (array): 
        ref_flux (array): 
    
    Returns:
        vmax (float): Velocity of the cross-correlation peak. Positive velocity
            means that observed spectrum is red-shifted with respect to the 
            reference spectrum.
         corrmax (float): Peak cross-correlation amplitude.
    """
    nwav = flux.size

    # Build spline object for resampling the model spectrum
    ref_spline = InterpolatedUnivariateSpline(ref_wav, ref_flux)

    # Convert target and model spectra to constant log-lambda scale
    wav, flux, dvel = loglambda(wav, flux)
    ref_flux = ref_spline(wav)

    # Perform cross-correlation, and use quadratic interpolation to
    # find the velocity value that maximizes the cross-correlation
    # amplitude. If `lag` is negative, the observed spectrum need to
    # be blue-shifted in order to line up with the observed
    # spectrum. Thus, to put the spectra on the same scale, the
    # observed spectrum must be red-shifted, i.e. vmax is positive.
    flux-=np.mean(flux)
    ref_flux-=np.mean(ref_flux)
    lag = np.arange(-nwav + 1, nwav) 
    dvel = -1.0 * lag * dvel
    corr = np.correlate(ref_flux, flux, mode='full')
    vmax, corrmax = quadratic_max(dvel, corr)

    if plot:
        from matplotlib import pylab as plt
        fig,axL = plt.subplots(ncols=2)
        plt.sca(axL[0])
        plt.plot(wav,ref_flux)
        plt.plot(wav,flux)
        plt.sca(axL[1])
        vrange = (-100,100)
        b = (dvel > vrange[0]) & (dvel < vrange[1])
        plt.plot(dvel[b],corr[b])
        plt.plot([vmax],[corrmax],'o',label='Cross-correlation Peak')
        fig.set_tight_layout(True)
        plt.draw()
        plt.show()

    return vmax, corrmax

def wav_to_dvel(wav):
    """Converts wavelengths to delta velocities using doppler formula"""
    dvel = (wav[1:] - wav[:-1]) / (wav[1:]) * SPEED_OF_LIGHT
    return dvel

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
    dvel = wav_to_dvel(wav)
    dvel = np.mean(dvel)
    return wav, flux, dvel

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
