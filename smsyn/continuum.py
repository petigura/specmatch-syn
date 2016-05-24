import numpy as np

def GPdt(xi,x,y,corrlen=20):
    """
    Gaussian Process-based detrending

    Same sigature as interp(xi,x,y).
    """

    def kernel(a, b):
        """
        GP squared exponential kernel
        """
        sqdist  = np.sum(a**2,1).reshape(-1,1) + \
                  np.sum(b**2,1) - \
                  2*np.dot(a, b.T)

        return np.exp( -.5 * sqdist / corrlen**2 )

    X  = x[:,np.newaxis]
    Xi = xi[:,np.newaxis]

    K  = kernel(X,X)
    s  = 0.05    # noise variance.
    N  = len(X)   # number of training points.
    L  = np.linalg.cholesky(K + s*np.eye(N))
    Lk = np.linalg.solve(L, kernel(X, Xi) )
    mu = np.dot(Lk.T, np.linalg.solve(L, y))
    return mu

def cfit(spec):
    """
    Continuum fit.
    
    Divide the spectrum up into n segments. Determine the 95
    percentile for each segment. Fit a GP through these points.

    Parameters
    ----------
    """
    
    wcen,sp = bin_percentile(spec,10)
    meansp = np.mean(sp) # Use as the mean value of the GP
    c = GPdt(spec['w'],wcen,sp-meansp,corrlen=20)+meansp
    return c

def bin_percentile(spec,n):
    """
    """
    p = 95 # Use 95 percentile
    f = lambda s : wcen_percentile(s,p)
    res = map(f, np.array_split(spec,n) )
    res = np.array(res)
    wcen,sp = res[:,0],res[:,1]
    return wcen,sp

def wcen_percentile(spec,p):
    """
    For each bin, return the specified percentile, and central wavelength
    """
    sp = np.percentile(spec['s'],95)
    w = spec['w']
    wcen = (w[0] + w[-1])/2
    return wcen,sp

