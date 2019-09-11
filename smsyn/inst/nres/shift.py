import numpy as np

import smsyn.echelle

NSEG = 6
PIXVELSHIFT_METHOD = 'spline'
NPIX_CLIP = 20 # Number of pixels to clip off at the ends of orders.


def shift(wav, flux, uflux, ref_wav, ref_flux, return_velocities=False):
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

        uflux (array): Array with shape `(nord, npix)` with
            uncertanties of flux measurement. This array "along for
            the ride" and receives the same shift.

        ref_flux (array): Array with shape `(nref_wav, )` with the
            continuum-normalized flux of reference spectrum.

        return_velocities (bool): return the velocity shift as well as the
            shifted spectrum

    Returns:
        flux_shift (array): Array with shape `(nref_wav, )` target spectrum
            shifted to the reference wavelength scale.
        uflux_shift (array): Array with shape `(nref_wav, )` of uncertainty
            array, also shifted to the reference wavelength scale

    """
    sl = slice(NPIX_CLIP,-NPIX_CLIP)
    ech = smsyn.echelle.Echelle(wav[:,sl], flux[:,sl], uflux[:,sl])
    pixmid, vshift = smsyn.echelle.vshift(ech, ref_wav, ref_flux, nseg=NSEG)
    pvs = smsyn.echelle.PixelVelocityShift(ech.nord, ech.npix, pixmid, vshift)
    dvel = pvs.caculate_dvel(method=PIXVELSHIFT_METHOD)
    ech_shift = smsyn.echelle.shift_orders(ech, ref_wav, dvel)
    flux_shift, uflux_shift = smsyn.echelle.flatten(ech_shift)

    if return_velocities:
        return flux_shift, uflux_shift, dvel
    else:
        return flux_shift, uflux_shift
