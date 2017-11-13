from cStringIO import StringIO as sio

import numpy as np
from numpy import ma
import pandas as pd
import h5py
from matplotlib.gridspec import GridSpec
from matplotlib import pylab as plt
from matplotlib.transforms import blended_transform_factory
# from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
from matplotlib.offsetbox import AnchoredText

from smsyn import results
from smsyn import smio
from smsyn import conf
from smsyn import results_scrape
from smsyn import telluric
from smsyn import restwav
from smsyn import ccs

panels_nbest = 10
annkw = dict(loc=3,frameon=False,prop=dict(name='monospace',size='small'))

k2str = {'teff':'Teff',
         'logg':'log(g)',
         'fe':'[Fe/H]',
         'vsini':'VsinI',
         'logchi':'log10(chi)'}

def panels(smres,nbest=10,how='chi',libpar=None):
    """Plot array of diagnostic plots.

    Parameters
    ----------
    smres : Results DataFrame containing chi2 figure of merit for every match
    nbest : Number of best matches to plot
    libpar : If we know the parameters of the object (e.g. it's a
             library spectrum) we can plot these too.

    """

    s = """\
xk     yk    
teff   logchi
logg   logchi
fe     logchi
vsini  logchi
teff   logg  
logg   fe    
teff   fe    
teff   vsini 
teff   logg  
logg   fe    
teff   fe    
teff   vsini 
"""

    fig, axL = plt.subplots(nrows=3,ncols=4,figsize=(14,10) )
    axL = axL.flatten()
    dfplot = pd.read_table(sio(s),sep='\s*',engine='python')
    dfplot['xL'] = dfplot['xk'].replace(k2str)
    dfplot['yL'] = dfplot['yk'].replace(k2str)
    dfplot['ax'] = axL

    smres = results.smres_add_chi(smres)
    smres['logchi'] = np.log10(smres.chi)

    if how=='chi':
        smresbest = smres.sort_values(by='chi')
    elif how=='close':
        d = smio.kbc_query(smres.targobs[0])
        targname = d['name']
        tpar = dict(lib.ix[targname])
        smres['close'] = close(tpar,smres)
        smresbest = smres.sort_values(by='close')

    smresbest = smresbest.iloc[:nbest]

    smptstyle = dict(marker='.',lw=0,ms=2,color='k')
    lgptstyle = dict(marker='+',mew=2,ms=6,lw=0,color='Tomato')

    plot_frame(dfplot,smres,**smptstyle)
    plot_frame(dfplot,smresbest,**lgptstyle)

    def edges(k):
        mnx = min(smresbest[k])
        mxx = max(smresbest[k])
        wx = mxx - mnx
        return mnx,mxx,wx

    def zoom(xk,yk):
        mnx,mxx,wx = edges(xk)
        mny,mxy,wy = edges(yk)

        plt.xlim(mnx-wx,mxx+wx)
        plt.ylim(mny-wy,mxy+wy)

    smpar = results.SMpars(smres) # parameters returned by SM
    smpar['targobs'] = smres.targobs.iloc[0]

    d = smio.kbc_query(smpar['targobs'])
    targname = d['name']

    smpar['targname'] = targname
    
    # Plot verticle lines at the location where SM thinks the parameters are
    def plotline(x):
        ax = x['ax']
        xk = x['xk']
        plt.sca(ax)
        trans = blended_transform_factory(ax.transData,ax.transAxes)
        plt.axvline(smpar[xk],ls='--')
        plt.text(smpar[xk],0.9,'SM',transform=trans)
        
        if libpar is not None:
            plt.axvline(libpar[xk])
            plt.text(libpar[xk],0.9,'LIB',transform=trans)

    dfplot.iloc[:3].apply(plotline,axis=1)

    def setlims(d):
        ax = d['ax']
        x = smresbest[d['xk']]
        y = smresbest[d['yk']]
        plt.sca(ax)
        plt.ylim(min(y)-0.1,max(y)+0.1)
    dfplot.iloc[:4].apply(setlims,axis=1)

    # Plot dots showing both the library positions and the SM-derived positions
    def plotres(d):
        plt.sca(d['ax'])
        x = smpar[d['xk']]
        y = smpar[d['yk']]
        tkw = dict(ha='center',va='center')
        plt.plot([x],[y],'oc',ms=15,mew=0)
        plt.text(x,y,'SM',**tkw)

        if libpar is not None:
            x = libpar[d['xk']]
            y = libpar[d['yk']]
            plt.plot([x],[y],'oc',ms=15,mew=0)
            plt.text(x,y,'LIB',**tkw)

    dfplot.iloc[[4,5,6,8,9,10]].apply(plotres,axis=1)

    def ann(d):
        plt.sca(d['ax'])
        for i in range(5):
            x = smresbest.iloc[i][d['xk']]
            y = smresbest.iloc[i][d['yk']]
            txt = plt.text(x,y,i,color='m',weight='bold',size=16,alpha=0.7)

    dfplot.iloc[8:].apply(ann,axis=1)
    
    for i in dfplot.index:
        if dfplot.ix[i,'xL']=='teff':
            plt.sca(axL[i])
            plt.setp(plt.gca().xaxis.get_ticklabels(),rotation=45)

    limd = {'teff' : (3000,7000),
            'logg' : (2,5.5),
            'fe'   : (-1.5,0.5),
            'vsini': (0,25)}

    def setlims(d):
        plt.sca(d['ax'])
        xk = d['xk']
        yk = d['yk']
        if limd.keys().count(xk)==1:
            plt.xlim(*limd[xk])
        if limd.keys().count(yk)==1:
            plt.ylim(*limd[yk])
    dfplot.apply(setlims,axis=1)


    def dfzoom(d):
        plt.sca(d['ax'])
        zoom(d['xk'],d['yk'])
    dfplot.iloc[8:].apply(dfzoom,axis=1)
    
    plt.sca(axL[8])

    flipd = {'teff' : True,
             'logg' : True,
             'fe'   : False,
             'vsini': False}

    def flipax(d):
        plt.sca(d['ax'])
        xk = d['xk']
        yk = d['yk']
        if flipd.keys().count(xk)==1:
            if flipd[xk]:
                flip('x')
        if flipd.keys().count(yk)==1:
            if flipd[yk]:
                flip('y')
    dfplot.apply(flipax,axis=1)
    flip('both')
    
    axL = plt.gcf().get_axes()
    for ax in axL:
        plt.sca(ax)
        if ax.get_xlabel()=='Teff':
            plt.setp(ax.xaxis.get_ticklabels(),rotation=20)
            
    stitle = "%(targname)-10s %(targobs)s\n" % smpar
    if libpar is not None:
        stitle +="LIB: %(teff)i %(logg).2f %(fe)+.2f\n" % libpar
    stitle +=" SM: %(teff)i %(logg).2f %(fe)+.2f " % smpar
             
    plt.sca(axL[0])
    plt.gcf().set_tight_layout(True)
    plt.title(stitle,fontdict=dict(family='monospace'))
    plt.draw()

def plot_frame(dfplot,df,**kwargs):
    """
    Plot DataFrame
    
    Helper function for panels.
    """
    for i in dfplot.index:
        x = dfplot.ix[i]
        plt.sca(x['ax'])
        plt.plot(df[x['xk']],df[x['yk']],**kwargs)
        plt.xlabel(x['xL'])
        plt.ylabel(x['yL'])

def flip(axis):
    if axis=='x':
        plt.xlim(plt.xlim()[::-1])
    if axis=='y':
        plt.ylim(plt.ylim()[::-1])
    if axis=='both':
        plt.xlim(plt.xlim()[::-1])        
        plt.ylim(plt.ylim()[::-1])


################################################
# Functions associated with plot_matches_group #
################################################
def annotate_matches(par):
    ax = plt.gca()
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    s = "%(libname)-10s %(teff)i %(logg).2f %(fe)+.2f %(chi)5.3g " % par
    plt.text(1.02,par['y'],s,transform=trans,family='monospace',color='m')

def close(par,lib):
    """
    How close are two spectra?

    Calculate distance between a given (Teff,logg,fe) triple to library values?
    
    Parameters
    ----------
    par : dictionary with the parameters of interest. Must contain the
          following keys
          - teff
          - logg
          - fe

    lib : DataFrame that contains library positions. Must contain
          following columns
          - teff
          - logg
          - fe      

    Returns
    -------
    dist : array with the distances to all the library points

    """
    
    par['eteff'] = 100
    par['elogg'] = 0.1
    par['efe'] = 0.1
    
    dist = np.zeros(len(lib))
    for k in 'teff logg fe'.split():
        dist+=((lib[k] - par[k])/par['e'+k])**2
    dist = np.sqrt(dist)
    return dist

def plot_matches_group(g,how='chi',ntop=8):
    """
    Plot matches from h5 group

    Pulls out the relavent arrays from a group and runs plot_matches
    
    Parameters
    ----------
    g : h5 group containing the following datasets
        - arr : DataSet with spectra
        - smres : DataSet with specmatch results
    
    """
    smres = pd.DataFrame(g['smres'][:])
    smres = results.smres_add_chi(smres)

    lspec = g['lspec'][:]
    smres.index = np.arange(len(smres))

    if how=='chi':
        smresbest = smres.sort_values(by='chi')
    elif how=='close':
        targname = smio.kbc_query(smres.targobs[0])['name']
        tpar = dict(lib.ix[targname])
        smres['close'] = close(tpar,smres)
        smresbest = smres.sort_values(by='close')
    smresbest = smresbest.iloc[:ntop]

    plot_matches(smresbest,lspec)
    plt.sca(plt.gcf().get_axes()[0])

def plot_matches(smresbest,lspec):
    """
    Plot best matches

    Plots the target spectrum along with the top matches
       
    """
    shift = 1
    fig,axL = plt.subplots(nrows=2,figsize=(20,12),sharex=True)
    
    plt.sca(axL[0])
    
    targpar = smresbest.rename(columns={'targobs':'obs'})
    targpar = dict(targpar['obs ord wlo whi'.split()].iloc[0])
    targpar['type'] = 'cps'
    targspec = smio.getspec_h5(**targpar)
    w = targspec['w']
    plt.plot(w,targspec['s'],'k')

    plt.rc('axes',color_cycle=['Tomato', 'RoyalBlue'])    
    for i in smresbest.index:
        # Plot target spectrum
        plt.sca(axL[0])
        y = shift*0.3
        plt.plot(w,lspec['lspec'][i]+y)

        par = dict(smresbest.ix[i])
        par['y'] = y+1
        annotate_matches(par)
        
        # Plot residuals
        plt.sca(axL[1])
        y = shift*0.2
        plt.plot(w,lspec['fres'][i]+y)

        par['y'] = y
        annotate_matches(par)
        shift+=1

    fig.subplots_adjust(left=.03,bottom=.03,top=0.97,right=.8)

def plot_quicklook(h5file):
    """
    Quick look plot

    Generate a 1-page quick look plot that captures the important
    spectral information
    """
    
    #    fig = figure()
    nrowtop = 2
    ncols = 5
    fig = plt.figure(figsize=(20,12))
    fig.set_tight_layout(True)

    nseg = len(conf.wloL_fm)

    gs = GridSpec(nseg+nrowtop+1,ncols)

    axTopL = [fig.add_subplot(gs[0:nrowtop,i]) for i in range(0,4)]
    axSpecFull = fig.add_subplot(gs[nrowtop,:])

    axSegL = [fig.add_subplot(gs[i,:]) for i in range(nrowtop+1,nrowtop+1+nseg)]

    h5 = h5py.File(h5file)

    for i in range(nseg):
        group = conf.wloL_fm[i]

        g = h5["polish_%i" % group]
        res = g['res'][:]
        w = res['w']
        tspec = res['mspec']+res['fres']
        mspec = res['mspec']
        mask = res['mask']
        res = res['res']

        kw = dict(lw=1,alpha=0.8)
        plt.sca(axSpecFull)
        plt.plot(w,mspec,color='Tomato',label='SpecMatch Best Fit',**kw)
        plt.plot(w,res,color='Tomato',**kw)
        plt.plot(w,tspec,color='k',label='Target Spectrum',**kw)
        axvspan_mask(w,mask)

        plt.sca(axSegL[i])
        plt.plot(w,mspec,color='Tomato',label='SpecMatch Best Fit',**kw)
        plt.plot(w,res,color='Tomato',**kw)
        plt.plot(w,tspec,color='k',label='Target Spectrum',**kw)
        axvspan_mask(w,mask)
        plt.ylim(-0.2,1.2)

    s = """\
xk     yk    
teff   logg  
fe     logg
"""

    dfplot = pd.read_table(sio(s),sep='\s*',engine='python')
    dfplot['xL'] = dfplot['xk'].replace(k2str)
    dfplot['yL'] = dfplot['yk'].replace(k2str)
    dfplot['ax'] = axTopL[:2]

    panelgroup = 6100
    smres = h5['%i' % panelgroup]['smres'][:]
    smres = pd.DataFrame(smres)
    smres['chi'] = smres.fchi
    smresbest = smres.sort_values(by='chi')
    smresbest = smresbest.iloc[:panels_nbest]

    smptstyle = dict(marker='.',lw=0,ms=2,color='k')
    lgptstyle = dict(marker='+',mew=2,ms=6,lw=0,color='Tomato')

    plot_frame(dfplot,smres,**smptstyle)
    plot_frame(dfplot,smresbest,**lgptstyle)

    for i in [0,2]:
        plt.setp(axTopL[i],xlim=(8000,3000),ylim=(6,0))
        plt.setp(axTopL[i].get_xticklabels(),rotation=20)

    plt.setp([axTopL[0],axTopL[1]],title='Best 10 matches wlo = %i' % panelgroup)
    plt.setp(axTopL[1],xlim=(-1.5,1),ylim=(6,0))

    plt.sca(axTopL[2])
    smpar = results_scrape.polish(h5file)
    x = smpar['teff']
    y = smpar['logg']
    tkw = dict(ha='center',va='center')
    plt.plot([x],[y],'oc',ms=15,mew=0)
    plt.text(x,y,'SM',**tkw)

#    isochrone.plotiso()

    plt.setp(plt.gca(),xlabel='Teff',ylabel='logg',title='Polished Parameters')


    plt.sca(axTopL[3])
    telluric.telluric_psf(smpar['obs'],plot_med=True)
    plt.title('Median Telluric Line Profile')

    s = """
Star    %(name)s
CPS ID  %(obs)s
Decker  %(DECKNAME)s

Paramaters (uncalibrated):
Teff   =  %(teff)i K
logg   =  %(logg).3f (cgs)
[Fe/H] =  %(fe).3f dex
VsinI  =  %(vsini).1f km/s
PSF    =  %(sig).1f pix
chi    =  %(chi).6f
""" % smpar
    plt.text(1.4,1,s,transform=plt.gca().transAxes,va='top',ha='left',family='monospace')


def plotspecmodel(w,tspec,mspec,res,mask):
    """
    Plot these three quantities
    
    tspec : target spectrum
    mspec : model spectrum
    res : residuals
    mask : wavelenth mask (plot as regions)
    """
    plt.plot(w,tspec)
    plt.plot(w,mspec)
    plt.plot(w,res)
    
    sL = ma.flatnotmasked_contiguous(ma.masked_array(mask,~mask))
    if sL is not None:
        for s in sL:
            plt.axvspan(w[s.start],w[s.stop-1],color='LightGrey')
    plt.ylim(-0.2,1.2)
    plt.xlim(min(w), max(w))


def axvspan_mask(x,mask):
    """
    Plot these three quantities
   
    x : independent variable
    mask : what ranges are masked out
    """
    
    sL = ma.flatnotmasked_contiguous(ma.masked_array(mask,~mask))
    if sL is not None:
        for s in sL:
            plt.axvspan(x[s.start],x[s.stop-1],color='LightGrey')


#########################################
# Functions associated with plot_polish #
#########################################
def stackax(axL):
    """
    Stack Axes
    
    Call a plotting function multiple times for every axis. Then
    adjust the xlims, so we see only a segment.
    """
    def wrap(f):
        nax = len(axL)
        def wrapped_f(*args):
            for ax in axL:                
                plt.sca(ax)
                f(*args)
            xl = plt.xlim()
            start = xl[0]
            step = (xl[1]-xl[0]) / float(nax)
            for ax in axL:                
                plt.sca(ax)
                plt.xlim(start,start+step)
                start+=step
        return wrapped_f
    return wrap

def axvspan_mask(x,mask):
    """
    Plot these three quantities
   
    x : independent variable
    mask : what ranges are masked out
    """
    
    sL = ma.flatnotmasked_contiguous(ma.masked_array(mask,~mask))
    if sL is not None:
        for s in sL:
            plt.axvspan(x[s.start],x[s.stop-1],color='LightGrey')

def plot_polish(h5file,group,libpar=None):
    """
    libpar : if parameters are known from some other technique, plot them.
    """
    h5 = h5py.File(h5file)
    g = h5['polish_%i' % group]
    res = g['res'][:]
    w = res['w']
    tspec = res['mspec']+res['fres']
    mspec = res['mspec']
    mask = res['mask']
    res = res['res']


    fig = plt.figure(figsize=(20,12))
    fig.set_tight_layout(True)
    gs = GridSpec(4,6)

    # Top row
    axLspec = [fig.add_subplot(gs[i,:4]) for i in range(1,4)]
    axLccf = [fig.add_subplot(gs[i,4]) for i in range(1,4)]
    axLacf = [fig.add_subplot(gs[i,5]) for i in range(1,4)]

    axccfsum = fig.add_subplot(gs[0,4]) 
    axacfsum = fig.add_subplot(gs[0,5])

    segdf = conf.segdf
    segdf.ix[int(group)]

    @stackax(axLspec)
    def diag_plot():
        axvspan_mask(w,mask)
        if libpar!=None:
            teff = libpar['teff']
            logg = libpar['logg']
            fe = libpar['fe']
            vsini = libpar['vsini']
            psf = libpar['psf']

            mspeclib = coelho.coelho_synth(teff,logg,fe,vsini,psf,**segdf.ix[int(group)])
            kw = dict(lw=2,alpha=0.8)
            plt.plot(w,mspeclib['s'],color='c',label='SpecMatch Synth Library Values',**kw)
            plt.plot(w,tspec-mspeclib['s'],color='c',**kw)
            plt.ylim(-0.5,1.5)

        kw = dict(lw=2,alpha=0.8)
        plt.plot(w,mspec,color='Tomato',label='SpecMatch Best Fit',**kw)
        plt.plot(w,res,color='Tomato',**kw)
        plt.plot(w,tspec,color='k',label='Target Spectrum',**kw)

    diag_plot()
    par = dict(h5['fm'].attrs)
    stitle = 'wlo = %i\n' %(group)
    stitle+= '%(teff)i %(logg).2f %(fe).2f %(vsini).2f (SpecMatch)\n' % par
    if libpar!=None:
        stitle+= '%(teff)i %(logg).2f %(fe).2f %(vsini).2f (Library)\n' % libpar

    plt.sca(axLspec[0])

    plt.title(stitle,family='monospace',size='medium',ha='left')
    
    bins = np.linspace(w[0],w[-1],4)
    binslo = bins[:-1]
    binshi = bins[1:]

    def plot_acf_ccf(axacf,axccf,wlo,whi):
        b = (w > wlo) & (w < whi)
        plt.sca(axccf)
        dvmax = plot_ccf(w[b], tspec[b], mspec[b])

        plt.sca(axacf)
        thwhm,mhwhm = plot_acf(w[b], tspec[b], mspec[b])
        return dvmax,thwhm,mhwhm

    res = map(plot_acf_ccf,axLacf,axLccf,binslo,binshi)
    res = np.array(res)

    plt.sca(axccfsum)
    plt.plot(res[:,0],label='Peak CCF (km/s)')

    plt.sca(axacfsum)
    plt.plot(res[:,1],label='Target HWHM (km/s)')
    plt.plot(res[:,2],label='Model HWHM (km/s)')

    plt.draw()
    fig.set_tight_layout(False)
    fig.subplots_adjust(top=0.95)

    plt.draw()    

def plot_polish_seg(h5file,group,libpar=None):
    """
    libpar : if parameters are known from some other technique, plot them.
    """
    h5 = h5py.File(h5file)
    g = h5[group]
    res = g['res'][:]
    w = res['w']
    tspec = res['mspec']+res['fres']
    mspec = res['mspec']
    mask = res['mask']
    res = res['res']


    fig = plt.figure(figsize=(20,12))
    fig.set_tight_layout(True)
    gs = GridSpec(4,6)

    # Top row
    axLspec = [fig.add_subplot(gs[i,:4]) for i in range(1,4)]
    axLccf = [fig.add_subplot(gs[i,4]) for i in range(1,4)]
    axLacf = [fig.add_subplot(gs[i,5]) for i in range(1,4)]

    axccfsum = fig.add_subplot(gs[0,4]) 
    axacfsum = fig.add_subplot(gs[0,5])

    segdf = conf.segdf
    wlo = int(group[-4:])
    segdf.ix[wlo]

    @stackax(axLspec)
    def diag_plot():
        axvspan_mask(w,mask)
        if libpar!=None:
            teff = libpar['teff']
            logg = libpar['logg']
            fe = libpar['fe']
            vsini = libpar['vsini']
            psf = libpar['psf']

            mspeclib = coelho.coelho_synth(teff,logg,fe,vsini,psf,**segdf.ix[int(group)])
            kw = dict(lw=2,alpha=0.8)
            plt.plot(w,mspeclib['s'],color='c',label='SpecMatch Synth Library Values',**kw)
            plt.plot(w,tspec-mspeclib['s'],color='c',**kw)
            plt.ylim(-0.5,1.5)

        kw = dict(lw=2,alpha=0.8)
        plt.plot(w,mspec,color='Tomato',label='SpecMatch Best Fit',**kw)
        plt.plot(w,res,color='Tomato',**kw)
        plt.plot(w,tspec,color='k',label='Target Spectrum',**kw)

    diag_plot()

    stitle = '%s wlo = %i\n' %(h5file,wlo)
#    stitle+= '%(teff)i %(logg).2f %(fe).2f %(vsini).2f (SpecMatch)\n' % dict(g.attrs)
#    if libpar!=None:
#        stitle+= '%(teff)i %(logg).2f %(fe).2f %(vsini).2f (Library)\n' % libpar
#
    plt.sca(axLspec[0])
    
    plt.title(stitle,family='monospace',size='medium',ha='left')
    
    bins = np.linspace(w[0],w[-1],4)
    binslo = bins[:-1]
    binshi = bins[1:]

    def plot_acf_ccf(axacf,axccf,wlo,whi):
        b = (w > wlo) & (w < whi)
        plt.sca(axccf)
        dvmax = plot_ccf(w[b], tspec[b], mspec[b])

        plt.sca(axacf)
        thwhm,mhwhm = plot_acf(w[b], tspec[b], mspec[b])
        return dvmax,thwhm,mhwhm

    res = map(plot_acf_ccf,axLacf,axLccf,binslo,binshi)
    res = np.array(res)

    plt.sca(axccfsum)
    plt.plot(res[:,0],label='Peak CCF (km/s)')

    plt.sca(axacfsum)
    plt.plot(res[:,1],label='Target HWHM (km/s)')
    plt.plot(res[:,2],label='Model HWHM (km/s)')

    plt.draw()
    fig.set_tight_layout(False)
    fig.subplots_adjust(top=0.95)

    plt.draw()    

def plot_ccf(w,tspec,mspec):
    lag,tmccf = ccf(tspec,mspec)

    dv = restwav.loglambda_wls_to_dv(w) 
    dv = lag*dv

    dvmax = ccs.findpeak(dv,tmccf)
    plt.plot(dv,tmccf,'k',label='tspec')    
    plt.axvline(dvmax,color='RoyalBlue',lw=2,alpha=0.4,zorder=0)

    AddAnchored("dv (max) = %.2f km/s" % dvmax,**annkw)
    plt.xlim(-50,50)
    plt.xlabel('dv (km/s)')

    return dvmax

def ccf(x,y):
    mx = np.mean(x)
    my = np.mean(y)
    ccf = np.correlate(x - mx, y - my, mode='full') 
    npix = x.size
    lag = np.arange(-npix+1,npix)
    return lag,ccf


def AddAnchored(*args,**kwargs):
    """
    Init definition: AnchoredText(self, s, loc, pad=0.4, borderpad=0.5, prop=None, **kwargs)
    Docstring:       AnchoredOffsetbox with Text
    Init docstring:
    *s* : string
    *loc* : location code
    *prop* : font property
    *pad* : pad between the text and the frame as fraction of the font
            size.
    *borderpad* : pad between the frame and the axes (or bbox_to_anchor).

    other keyword parameters of AnchoredOffsetbox are also allowed.
    """

    at = AnchoredText(*args,**kwargs)
    plt.gca().add_artist(at)

def plot_acf(w,tspec,mspec):
    lag,tacf = ccf(tspec,tspec)
    lag,macf = ccf(mspec,mspec)
    dv = restwav.loglambda_wls_to_dv(w) 
    dv = lag*dv
    plt.plot(dv,tacf,'k',label='tspec')
    plt.plot(dv,macf,'Tomato',label='mspec')
    
    # Label shifts that can include noise
    b = abs(lag)<=2 
    plt.plot(dv[b],tacf[b],'.',mec='none',mfc='k')
    plt.plot(dv[b],macf[b],'.',mec='none',mfc='Tomato')
    
    thwhm = hwhm(lag,tacf)
    mhwhm = hwhm(lag,macf)
    s = """\
HWHM = %.2f (targ)
HWHM = %.2f (model)""" % (thwhm,mhwhm)
    AddAnchored(s,**annkw)
    plt.xlim(-50,50)
    plt.xlabel('dv (km/s)')
    return thwhm,mhwhm

def hwhm(lag,acf):
    """
    Half-width at Half-max.

    Reads in h5 file and returns the hwhm information.

    Parameters
    ----------
    lag : displacement

    Returns
    -------
    hwhm : half-width at half max
    """
    lagint,acfint = peakIntrp(lag,acf)
    hm = np.max(acfint) / 2

    # Interpolate to find the half-width at half-max
    blagpos = lag > 0
    lagpos  = lag[blagpos]
    acfpos  = acf[blagpos]

    idmin = np.argsort(abs(acfpos-hm))[0]
    interpslice = slice(idmin-1,idmin+1) # choose closet 3

    lagp    = lagpos[interpslice]
    acfp = acfpos[interpslice]
    idsort = np.argsort(acfp)
    acfp = acfp[idsort]
    lagp = lagp[idsort]

    hwhm   = np.interp(hm,acfp,lagp)
    return hwhm

def peakIntrp(dv,fconv):
    """
    Peak interpolation

    Confluence of noise can produce a peak at displacements of less
    than 1 pixel. This little function fits a quadratic to the region
    near the peak.
    """
    adv = np.abs(dv)
    bfit = (adv < 5) & (adv > 2)
    xfit = dv[bfit]
    yfit = fconv[bfit]
    p1 = np.polyfit(xfit,yfit,2)
    bint = adv < 2 
    dvint = dv[bint]
    fconvint = np.polyval(p1,dvint)
    return dvint,fconvint
