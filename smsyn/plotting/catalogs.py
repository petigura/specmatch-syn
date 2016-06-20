"""
Catalog Plots

Plots for comparing SpecMatch catalogs to other library catalogs
"""
import isochrones
import os
import pandas as pd
import numpy as np
from matplotlib.pylab import *
from . import plotplus

k2str = {'teff':'Teff',
         'logg':'log(g)',
         'fe':'[Fe/H]',
         'vsini':'VsinI',
         'logchi':'log10(chi)'}

k2tex = {'teff':'$\mathrm{T}_{\mathrm{eff}}$',
         'logg':'$\log\ g$',
         'fe':'[Fe/H]',
         'vsini':'$v \sin i$',
         'logchi':'log10(chi)'}
units = dict(teff='K',logg='cgs',fe='dex')

def format(d,precision='low'):
    """
    Take a dictionary of numbers and convert to strings with the proper precision.
    """

    outd = {}
    for k,x in d.iteritems():
        if precision=='low':
            if k.count('teff')!=0:
                outd[k] = "%+i" % int(x)
            elif k.count('logg')!=0:
                outd[k] = "%+.3f" % x
            elif k.count('fe')!=0:
                outd[k] = "%+.2f" % x
            elif k.count('vsini')!=0:
                outd[k] = "%+.1f" % x
            else:
                outd[k] = str(k)
        else:
            if k.count('teff')!=0:
                outd[k] = "%i" % int(x)
            elif k.count('logg')!=0:
                outd[k] = "%.2f" % x
            elif k.count('fe')!=0:
                outd[k] = "%.2f" % x
            elif k.count('vsini')!=0:
                outd[k] = "%.1f" % x
            else:
                outd[k] = str(k)
    return outd

def to_tex(path,texcmd,s):
    """
    1. Open file at path
    2. Search for texcmd
    3. If texcmd exists, overwrite. Not, create new
    4. Close file
    """
    df = pd.read_table(path,names=['line'])
    line = '\\nc{\\%s}{%s}' % (texcmd,s)
    b = df.line.str.contains(texcmd)
    if b.sum()==0:
        df = pd.concat([df,pd.DataFrame([line],columns=['line'])])
    elif b.sum()==1:
        df.ix[df.index[b],'line'] = line
    else:
        print "multiple ref"
    df.to_csv(path,index=False,sep=' ',header=False)

def diffs(comb,axL=None,suffixes=['_lib','_sm'],**kw):
    """
    Plot the differences
    """

    if axL is None:
        fig,axL = subplots(nrows=3,ncols=1,figsize=(6,8))
        axL = axL.flatten()

    row = 0
    for k in 'teff logg fe'.split():
        sca(axL[row])
        gca().grid(True)

        kdiff = k+'_diff'
        d = comb[kdiff]

        k0 = k+suffixes[0]

        scatter(comb[k0],d,**kw)
        xlabel('%s (%s) [lib]' % (k2tex[k],units[k]) )
        ylabel('$\Delta$ %s (%s)' % (k2tex[k],units[k]) )


        sd = diffstats(d)
        sdkeys = sd.keys()
        sdkeys.remove('nout')
        sd = dict( [(kk+'_'+k,sd[kk]) for kk in sdkeys] )
        sd = format(sd)
        sd = dict( [(kk,sd[kk+'_'+k]) for kk in sdkeys])

        mode = 'clean'
        if mode=='full':
            s = """\
full  sigclip (%(nout)i)
diff      %(meandiff)s %(clipmeandiff)s
RMS(diff) %(disp)s  %(clipdisp)s""" % sd 
        elif mode=='clean':
            s = """\
Mean(Diff) %(meandiff)s
RMS(Diff)  %(disp)s  """ % sd 

        plotplus.AddAnchored(s,3,frameon=False,prop=dict(family='monospace',size='small',alpha=0.9))


        row+=1
    gcf().set_tight_layout(True)

def plotdiffs(df,xk,yk,suffixes=['_lib','_sm']):
    """
    Plot library values and SpecMatch values across the HR diagram 

    df : DataFrame with keys like
         teff_sm
         teff_lib
    """
    kx0 = xk+suffixes[0]
    ky0 = yk+suffixes[0]

    kx1 = xk+suffixes[1]
    ky1 = yk+suffixes[1]

    plot(df[kx0],df[ky0],'ok',ms=4,label='Library value')

    dy = df[[ky1,ky0]]
    dx = df[[kx1,kx0]]

    plot(dx.T,dy.T,'r')
    plot(dx.iloc[0],dy.iloc[0],'r',label='SM-derived value')

#    xlabel("%s (%s)" % (k2tex[xk],units[xk]))
#    ylabel("%s (%s)" % (k2tex[yk],units[yk]))



def merge_sm_lib(dfsm,dflib,suffixes=['_sm','_lib']):
    """
    Merge two catalog based on CPS observation number.
    """

    try:
        comb = pd.merge(dfsm,dflib.drop('name',axis=1),on=['obs'],
                        suffixes=suffixes)
    except:
        comb = pd.merge(dfsm,dflib,on=['name'],
                        suffixes=suffixes)

    s0 = suffixes[0]
    s1 = suffixes[1]

    csm = [c.split('_')[0] for c in comb.columns if c.count(s0)>0]
    print csm
    for c in csm:
        print c
        comb['%s_diff' % c] = comb["%s%s" % (c,s0)] - comb['%s%s' % (c,s1)]
    comb = comb[np.sort(list(comb.columns))]
    return comb 



def twopane(df,axL=None,suffixes=['_lib','_sm']):
    """
    Four pane diagnositic plots.

    Parameters
    ----------
    df : DataFrame with the results of SpecMatch. Must contain:
    """

    def tup(k):
        return [df[k+suffixes[0]],df[k+suffixes[1]]]

    if axL is None:
        fig,axL = subplots(ncols=2,figsize=(12,4),sharey=True)
        axL = axL.flatten()

    sca(axL[0])
    plotdiffs(df,'teff','logg',suffixes=suffixes)
    ylim(1.,5.0)
#    isochrone.plotiso()

    for l in gca().lines:
        label = l.get_label()
        if label=='Library value':
            label=''
        if label=='SM-derived value':
            label=''
        l.set_label(label)

        
#    gca().lines.remove(line)
    legend(fontsize='x-small',ncol=2,numpoints=3,title='Isochrones: Age, [Fe/H]')
    plotplus.flip('both')

    sca(axL[1])
    plotdiffs(df,'fe','logg',suffixes=suffixes)
    legend(fontsize='x-small')

def diffstats(d,mode='clean'):
    """
    Difference Statistics
    """
    mad = np.median(np.abs(d))
    b = np.abs(d) < 5*mad
    sd = dict(disp=std(d),nout=(~b).sum(),clipdisp=std(d[b]),
              meandiff=np.mean(d),clipmeandiff=np.mean(d[b]))
    return sd


def fivepane(comb,suffixes=['_lib','_sm'], color='fe'):
    """
    Five panel representation of the differences between two samples.
    """

    fig = figure(figsize=(7.5,6.5))

    gs = GridSpec(6,2)

    axLtwopane = [fig.add_subplot(gs[3*i:3*i+3,0]) for i in range(2)]
    axLdiffs = [fig.add_subplot(gs[2*i:2*i+2,1]) for i in range(3)]

    twopane(comb,axL=axLtwopane,suffixes=suffixes)
    sca(axLtwopane[0])
    xlim(7000,4000)
    sca(axLtwopane[1])
    plotplus.flip('y')

    c = comb[color+suffixes[0]]
    diffs(comb,axL=axLdiffs,c=c,linewidths=0,suffixes=suffixes)

    sca(axLdiffs[0])
    ylim(-300,300)
    xlim(4000,7000)

    sca(axLdiffs[1])
    ylim(-0.5,0.5)
    xlim(2.0,5.0)

    sca(axLdiffs[2])
    ylim(-0.3,0.3)
    xlim(-1.0,0.5)
    
    for ax,panelname in zip(gcf().get_axes(),'ABCDE'):
        sca(ax)
        plotplus.AddAnchored(panelname,2,prop={})

def axreplace(namemap):
    axL = gcf().get_axes()


    for ax in axL:
        sca(ax)
        for k in namemap.keys():
            xlabel(ax.get_xlabel().replace(k,namemap[k]))
            ylabel(ax.get_ylabel().replace(k,namemap[k]))




def plot_cks(df0,isokw={}):
    df = df0.copy()
    fig,axL = subplots(nrows=2,ncols=2,figsize=(8,8),sharex=True,sharey=True)

    for ax,panelname in zip(axL.flatten(),'ABCD'):
        sca(ax)
        isochrone.plotiso(**isokw)
        setp(ax.xaxis.get_ticklabels(),rotation=20)
        AddAnchored(panelname,1,prop={})

    sca(axL[1,0])
    xlabel('Teff')
    ylabel('logg')
    xlim(700)

    sca(axL[0,0])
    kwpass = dict(marker='.',ms=3,lw=0)
    kwfail = dict(marker='.',ms=5,lw=0,color='Tomato')

    sca(axL[0,0])
    plot(df.teff,df.logg,**kwpass)
    title("%i CKS Stars" % len(df),)

    def plot_pane(df,bpass):
        dffail = df[~bpass]
        dfpass = df[bpass]

        getn = lambda df : len(df[df.teff.notnull() & df.teff.notnull()])
        nfail = getn(dffail)
        npass = getn(dfpass)

        kwfail['label'] = "%i stars removed" % nfail
        kwpass['label'] = "%i stars remain" % npass

        plot(dffail.teff,dffail.logg,**kwfail)
        plot(dfpass.teff,dfpass.logg,**kwpass)        
        legend()
        return dfpass
    
    for k in 'bbinary bvsini bteff'.split():
        df[k] = df[k].astype(bool)
        
    sca(axL[0,1])
    df = plot_pane(df,~df.bbinary)
    setp(gca(),title="Removing Spectroscopic Binaries")
    setp(gca().get_title(),)

    sca(axL[1,0])
    df = plot_pane(df,df.bvsini)
    setp(gca(),title="Removing Stars with VsinI > 20 km/s")

    sca(axL[1,1])
    df = plot_pane(df,df.bteff)
    setp(gca(),title="Removing Stars with\nTeff > 7000 K or Teff < 4600 K (KIC)")

    gcf().set_tight_layout(True)
    xlim(4000,7000)
    ylim(1,5)
    flip('both')
