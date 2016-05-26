import numpy as np
from numpy import ma
from scipy.interpolate import InterpolatedUnivariateSpline
from . import wavsol
from .wavsol import SPEED_OF_LIGHT

class Echelle(object):
    """Echelle spectrum

    Args:
        wav (array): Array with shape `(nord, npix)` with wavelengths
            of observed spectrum. Usually determined from calibration
            lamps. This does not need to be too accurate because
            spectrum will be shifted to reference spectrum wavelength
            scale.

        flux (array): Array with shape `(nord, npix)` with continuum
            normalized flux of target spectrum.

        uflux (Optional[array]): Array with shape `(nord, npix)` with
            uncertanties of flux measurement. This array "along for
            the ride" and receives the same shift.
    """

    def __init__(self, wav, flux, uflux):
        assert wav.shape==flux.shape, "wav and flux must have same shape"
        self.nord, self.npix = wav.shape
        self.wav = wav
        self.flux = flux
        self.uflux = uflux

class PixelVelocityShift(object):
    """Pixel Velocity Shift

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

    def caculate_dvel(self, method, **kwargs):
        """Calculate pixel-by-pixel velocity shift

        Args:
            method (str): Method by which we calculate velocity shifts. Varies by
                spectrometer
            **kwargs: Passed to method that performs computation e.g.
                `_caculate_dvel_global()`

        Returns:
            dvel (array): Array with shape `(nord, npix)` with the velocity shifts
                at each pixel value
        """
        if method=='global':
            dvel = self._calculate_dvel_global(**kwargs)
            
        print "Solving for pixel-by-pixel velocity shift"
        print_vshift(self.pixmid, dvel[:,self.pixmid] )
        return dvel
        
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
        bout = np.abs(vshift - med) > sigclip * mad
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

def vshift(ech, ref_wav, ref_flux, nseg=8):
    """Compute velocity shift between echelle spectrum and a reference spectrum

    Given an extracted echelle spectrum, compute velocity displacement between it
    and a reference spectrum.

        ech (Echelle object): Echelle object having `nord` orders and
            `npix` measurements per order

        ref_wav (array): Array with shape `(nref_wav, )` of wavelengths of 
            reference spectrum.

        ref_flux (array): Array with shape `(nref_wav, )` with the
            continuum-normalized flux of reference spectrum.

        nseg (int): Number of segments to break each order into when 
            performing the cross-correlation.

    Returns:
        vshift (array): `nord `
        flux_shift (array): Array with shape `(nref_wav, )` target spectrum
            shifted to the reference wavelength scale.
        uflux_shift (array): Array with shape `(nref_wav, )` of uncertainty
            array, also shifted to the reference wavelength scale

    """
    assert ref_wav.shape==ref_flux.shape,\
        "ref_wav and ref_flux mush have same shape"

    nref_wav = ref_flux.shape[0]
    vshift = np.zeros( (ech.nord, nseg) )

    # array indecies associated with different segments 
    pixsegs = np.array_split(np.arange(ech.npix), nseg)
    pixmid = np.array([int(np.mean(x)) for x in pixsegs])

    print "Calculating velocity shifts order by order"
    for i_order in range(ech.nord):
        for i_seg in range(nseg):
            i_pix = pixsegs[i_seg]
            wav = ech.wav[i_order,i_pix]
            flux = ech.flux[i_order,i_pix]
            _vmax,_corrmax = wavsol.velocityshift(wav, flux, ref_wav, ref_flux)
            vshift[i_order,i_seg] = _vmax

    print_vshift(pixmid, vshift)
    return pixmid, vshift

def print_vshift(pixmid, vshift):
    nord = vshift.shape[0]
    outstr = ["{:7d}".format(v) for v in pixmid]
    outstr = " ".join(outstr)
    print "pixmid   " + outstr
    for i_order in range(nord):
        outstr = ["{:+7.2f}".format(v) for v in vshift[i_order]]
        outstr = " ".join(outstr)
        outstr = "order {:2d} ".format(i_order) + outstr
        print outstr

def shift_orders(ech, ref_wav, dvel):
    """Shift echelle orders

    Args:
        ech (Echelle object): Echelle object.
        ref_wav (array): reference wavelength with shape `(nref_wav, )`.
        dvel (array): Array with shape `(ech.nord, ech.npix)` with velocity shifts
            at each pixel.

    Returns:
        ech_shift (Echelle object): Echelle object with shape 
            `(ech.nord, nref_wav)`
    """

    # Create output echelle object
    nref_wav = ref_wav.shape[0]
    wav_shift = np.empty((ech.nord, nref_wav))
    flux_shift = np.empty((ech.nord, nref_wav))
    uflux_shift = np.empty((ech.nord, nref_wav))
    wav_shift[:] = ref_wav[np.newaxis,:]
    flux_shift[:] = np.nan
    uflux_shift[:] = np.nan
    ech_shift = Echelle(wav_shift, flux_shift, uflux_shift )
    
    for i_order in range(ech.nord):
        # Calculate the change in wavelength to the model Change
        # wavelengths to the rest wavelengths
        dlam = dvel[i_order] / SPEED_OF_LIGHT * ech.wav[i_order]
        wav_refscale = ech.wav[i_order] - dlam 
        b = (wav_refscale[0] < ref_wav) & (ref_wav < wav_refscale[-1])

        # Shift flux.
        spline = InterpolatedUnivariateSpline(wav_refscale, ech.flux[i_order])
        ech_shift.flux[i_order,b] = spline(ref_wav[b])

        # Shift uflux values.
        spline = InterpolatedUnivariateSpline(wav_refscale, ech.uflux[i_order])
        ech_shift.uflux[i_order,b] = spline(ref_wav[b])
    return ech_shift

def flatten(ech, method='average'):
    """Flatten 2-D echelle spectrum to 1-D flat spectrum
    """
    wav = ech.wav[0]
    assert np.allclose(ech.wav - wav, 0), "ech.wav rows must be identical"

    ech.flux = ma.masked_invalid(ech.flux)
    ech.uflux = ma.masked_invalid(ech.uflux)
    
    if method=='average':
        ivar = ech.uflux**-2
        # Weighted mean and uncertanty on weighted mean
        flux = ma.sum( ech.flux * ivar, axis=0 ) / ma.sum(ivar, axis=0)
        uflux = ma.sqrt( 1 / ma.sum(ivar, axis=0) )

    flux.fill_value = np.nan 
    uflux.fill_value = np.nan 
    flux = flux.filled()
    uflux = uflux.filled()
    return flux, uflux
