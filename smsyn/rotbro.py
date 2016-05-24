import numpy as np
from scipy import interpolate 
from scipy.io.idl import readsav

def rotbro(dw, s, lcen, vsini, eps=0.6, nres=10):
    """
    Rotational Broadening

    Parameters
    ----------
    dw    : array relative wavelength scale for spectrum to be broadened
    s     : array spectrum to be broadened
    lcen  : central wavelength of line (in Angstroms).
    vsini : equatorial velocity of star (km/sec)
    eps   : linear limb darkening coefficient
    nres  : number of points in broadening kernal
   
    Returns
    -------
    Spectrum with covolved with broadening kernel

    History
    -------
    Adapted by Erik Petigura IDL routine ROTBRO by Jeff Valenti
    """
    nresi = long(nres)
    c =  2.9979246e5

    # Return input spectrum if vsini is negative or zero.
    if vsini < 0.0:
        return s
    elif vsini < 0.4:
        vsini = 0.2 + 0.5*vsini #ramp to 0.2 ,lowest vsini

    dwmax = lcen * float(vsini) / c # maximum doppler shift (limb)

    # Generate Kernel
    ker = rotkern(dwmax,eps,nres)

    # Spline spectrum onto convolution wavelength scale.
    dx = dwmax / nresi                       
    nx = int((dw[-1] - dw[0] )/ dx + 1)      
    x = dw[0] + dx * np.arange(nx)           
    tck = interpolate.splrep(dw,s)
    y   = interpolate.splev(x,tck)

    # Convolve with broadening kernel.
    # First pad out reduce edge effects
    npad = nresi + 2                          
    npad = 3 * npad
    y = np.hstack([np.zeros(npad) + y[0],y,np.zeros(npad) + y[-1]])

    yout = np.convolve(y,ker,mode='same')  
    yout = yout[npad:-npad] # Clip off edge padding

    # (Cubic) spline back onto original wavelength scale.
    tck   = interpolate.splrep(x,yout)
    sout  = interpolate.splev(dw,tck)
    return sout

def rotkern(dwmax,eps,nres):
    """
    Rotational broadening kernel

    See Gray, _Photospheres_, p.393 or Gray, _Photospheres 2ed_, p.374

    # Note: dwmax is assumed constant over the wavelength range of W.
    """

    # constants of kernel function
    c1 = 2.0 * (1.0 - eps) / (np.pi * dwmax * (1.0 - eps/3.0))
    c2 = eps / (2.0 * dwmax * (1.0 - eps/3.0))	

    # fraction of max change in W
    dwfrac = np.arange(2*nres + 1,dtype=np.float) / nres - 1.0 

    # Return normalized broadening kernel
    z   = 1.0 - dwfrac * dwfrac 
    ker = c1 * np.sqrt(z) + c2 * z
    ker = ker / np.sum(ker)      
    return ker
