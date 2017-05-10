"""
Module to create some useful plots from the HIRES pipeline
"""
from matplotlib.pylab import *

def plot_fit(pipe, segment0, method):
    row = np.where(pipe.segments[:,0]==segment0)[0]
    assert len(row)==1,"invalid segment"
    segment = pipe.segments[row[0]]
    nrows = 3 
    if method=='polish':
        output = pipe.polish_output[segment0]
    if method=='lincomb':
        output = pipe.lincomb_output[segment0]

    fig, axL = subplots(nrows=nrows,figsize=(8,6))
    xlims = linspace(segment[0],segment[1],nrows+1)


    wav_exclude = pipe.wav_exclude
    wav = output['wav'] 
    flux = output['flux']
    resid = output['resid']
    wavmask = output['wavmask']
    model = flux - resid

    # Identifies the contiguous masked regions
    sL = ma.notmasked_contiguous(ma.masked_array(wavmask,~wavmask))
    wav_exclude = [ (wav[slice.start],wav[slice.stop-1]) for slice in sL]

    for i in range(nrows):
        sca(axL[i])
        plot(wav, flux,'k')
        plot(wav, model, color='Tomato')
        plot(wav, resid, color='RoyalBlue')
        xlim(xlims[i],xlims[i+1])

        if sL is None:
            continue

        for wavex in wav_exclude:
            axvspan(wavex[0],wavex[1],color='LightGray')

    fig.set_tight_layout(True)
    setp(axL[-1],xlabel='Wavelength',ylabel='Intensity')
    setp(axL, ylim=(-0.2,1.2) ) 
