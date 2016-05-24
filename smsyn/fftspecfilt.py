import numpy as np
from numpy.fft import fft,ifft,fftfreq

def fftspecfilt(spec,filtpar):
    """
    high-pass filters a 1-D spectrum 

    using FFTs with a cutoff frequency given by 1 / filtpar returns the
    high-frequency components the low-frequency components can be
    returned using the lowpass keyword

    Parameters
    ----------
    spec : spectrum
    filtpar : remove variations longer than this amount.

    History
    -------
    2013dec02 : Adaped by Erik Petigura from Andrew Howard's IDL  version

    """

    npix = spec.size
    odd  = (npix % 2 == 1) # True if npix is odd

    if odd:        
        y = np.hstack([np.arange(npix/2+1),np.arange(npix/2)-npix/2])
        y[npix/2+1:npix-1] = y[0:npix/2-1][::-1]
    else:
        y = np.hstack([np.arange(npix/2),np.arange(npix/2)-npix/2])
        y[npix/2:npix-1] = y[0:npix/2-1][::-1]

    filter = 1.0/(1+(y/filtpar)**10)
    out = ifft(fft(spec)*(1.0-filter)) # highpass filtered
    out = out.real
    return out

def fftbandfilt(spec,wlo=None,whi=None):
    """

    Parameters
    ----------
    spec : spectrum
    filtpar : remove variations longer than this amount.

    History
    -------
    2013dec02 : Adaped by Erik Petigura from Andrew Howard's IDL  version

    """

    fspec = fft(spec) # Fourier transform of spectrum
    freq = fftfreq(spec.size,d=1.0) # Cycles per pixel
    
    if wlo==None:
        fhi = max(freq)
    else:
        fhi = 1./wlo

    if whi==None:
        flo = min(freq)
    else:
        flo = 1./whi

    fspec[ (abs(freq) > fhi) | (abs(freq) < flo) ] = 0.
    filtspec = ifft(fspec).real

    return filtspec
