"""
Module to create some useful plots from the HIRES pipeline
"""
from matplotlib.pylab import *

def plot_fit(pipe,segment0,method):
    row = np.where(pipe.segments[:,0]==segment0)[0]
    assert len(row)==1,"invalid segment"
    segment = pipe.segments[row[0]]
    print segment
    nrows = 3 
    if method=='polish':
        output = pipe.polish_output[segment0]
    if method=='lincomb':
        output = pipe.lincomb_output[segment0]

    print "reduced chisq = {rchisq:.2f}".format(**output)
    fig,axL = subplots(nrows=nrows,figsize=(8,6))
    xlims = linspace(segment[0],segment[1],nrows+1)
    wav_exclude = pipe.wav_exclude
    for i in range(nrows):
        sca(axL[i])

        model = output['flux'] - output['resid']
        plot(output['wav'], output['flux'],'k')
        plot(output['wav'], model, color='Tomato')
        plot(output['wav'], output['resid'], color='RoyalBlue')
        xlim(xlims[i],xlims[i+1])

        for irow in range(wav_exclude.shape[0]):
            axvspan(wav_exclude[irow][0],wav_exclude[irow][1],color='LightGray')

    fig.set_tight_layout(True)
    setp(axL[-1],xlabel='Wavelength',ylabel='Intensity') 
