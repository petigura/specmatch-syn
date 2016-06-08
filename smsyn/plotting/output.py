"""
"""
from matplotlib import pylab as plt

def chisq(df, fig=None, columns=['teff','logg','fe'], **kwargs):
    """Make a multi-panel plot of chisq
    """
    ncols = len(columns)
    if fig is None:
        fig,axL = plt.subplots(ncols=ncols)
    else:
        axL = fig.get_axes()

    i = 0
    for col in columns:
        plt.sca(axL[i])
        plt.plot(df[col],df['chisq'],**kwargs)
        i+=1

    
