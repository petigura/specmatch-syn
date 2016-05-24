import numpy as np
from scipy import interpolate
from scipy.io.idl import readsav

def ccs(obslambda,obsraw,tlambda,traw,r=[-200,200]):
    """
    CCS Cross-correlate spectra

    Parameters
    ----------
    obslambda  : observed wavelength
    obsraw     : observed intensity
    tlamba     : template wavelength
    traw       : template intensity
    r          : range of lags to try in the cross-correlation. convscale puts
                 the wavelength onto a constant log(lambda) wavelength
                 scale. Each lag correspons to delvel from convscale/

    Returns
    -------
    velm : shift in wavelength

    """

    delvel,o = convscale(obslambda,obsraw)
    delvel,t = convscale(tlambda,traw)

    cf  = np.correlate(t-1,o-1,mode='full')

    from matplotlib.pylab import *
    clf()
    plot(o)
    plot(t)
    #import pdb;pdb.set_trace()
    
    npix = o.size
    lag  = np.arange(-npix+1,npix)

    s   = slice(npix + r[0],npix + r[1]+1 )
    vel = findpeak(lag[s],cf[s])

    velm = vel*delvel
    velm *=-1 # Hack to make it agree with IDL version.
    return velm
    
def findpeak(x,y,r=3,ymax=False):
    """
    Fit a parabola near a maximum, and figure out where the derivative is 0.
    """
    idmax = np.argmax(y)
    idpeak = np.arange(idmax-r,idmax+r+1)
    a = np.polyfit(x[idpeak],y[idpeak],2)
    xmax = -a[1]/(2.*a[0]) # Where the derivative is 0

    if ymax!=False:
        ymax = np.polyval(a,xmax)
        return xmax,ymax
    else:
        return xmax

