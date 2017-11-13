"""
Plots to summarize results
"""

import os

from matplotlib import pylab as plt
import pylab as pl
import numpy as np
import pandas as pd

import isochrones
import isochrones.mist

import smsyn.library
import smsyn.wavsol
import smsyn.plotting.utils

c = smsyn.wavsol.SPEED_OF_LIGHT

mist = isochrones.mist.MIST_Isochrone()


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
        plt.semilogy()
        plt.plot(df[col],df['chisq'],**kwargs)
        i+=1


def ACF(x):
    mx = np.mean(x)
    acf = np.correlate(x - mx, x - mx, mode='full')
    npix = x.size
    lag = np.arange(-npix+1,npix)
    return lag,acf


def CCF(x,x2):
    mx = np.mean(x)
    mx2 = np.mean(x2)
    ccf = np.correlate(x - mx, x2 - mx2, mode='full')
    npix = x.size
    lag = np.arange(-npix+1,npix)
    return lag,ccf


def plotHR(libfile, teff, logg):
    """
    Plot library values and SpecMatch values across the HR diagram

    Args:
        libfile (string): path to cataloge of reference stars
        teff (float): effective temperature of target
        logg (float): logg of target

    """
    lib = pd.read_csv(libfile)

    pl.plot(lib.TEFF, lib.LOGG, 'k.', color='0.6', label='SM Library')

    pl.plot(teff, logg, 'r*', ms=30, label='Library value', markeredgecolor='k', markeredgewidth=2)

    ax = pl.gca()

    pl.ylim(2.5, 5)
    pl.xlim(*pl.xlim()[::-1])
    pl.ylim(*pl.ylim()[::-1])
    pl.xticks(pl.xticks()[0][1::2], fontsize=18)
    pl.xlabel('$T_{\\rm eff}$', fontsize=20, labelpad=-5)
    pl.ylabel('$\\log{g}$', fontsize=20)

    ax.tick_params(pad=5)

    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    pl.yticks(fontsize=18)


def plotiso():

    feh = [-0.5, 0, 0.5]
    age = np.log10(5e9)
    for fe in feh:

        iso = isochrones.Isochrone.isochrone(mist, age, feh=fe)

        pl.plot(iso['Teff'], iso['logg'], color='RoyalBlue', ls='-', lw=1)


def loglambda_wls_to_dv(w, nocheck=True):
    """
    Checks that spectrum is on a constant loglambda wavelength
    scale. Then, returns dv
    """
    dv = (w[1:] - w[:-1])/(w[1:])*c
    mdv = np.mean(dv)
    if not nocheck: assert dv.ptp() / mdv < 1e-6,"wavelengths must be log-lambda"
    dv = mdv
    return dv


def bestfit(bestpars, pipe, title, method='polish', outfile='bestfit.pdf', *args, **kwargs):

    # segs = np.array([[5210,5240], [5360,5375], [5520,5540]])
    # segs = np.array([[5160,5190], [5880,5910], [5940,6000]])     # New APF
    # segs = np.array([[5220, 5250], [5545, 5575], [6120, 6150]])  # APF+Keck compatable
    # segs = np.array([[5220,5250], [5407,5437], [6120,6150]])     # APF+Keck compatable
    # segs = np.array([[5160,5190], [5277,5307], [5880,5910]])     # interesting, but not well-fit
    segs = np.array([[5160, 5190], [5545, 5575], [6120, 6150]])    # include Mg region
    # segs = np.array([[5250, 5280], [5545, 5575], [6120, 6150]])    # include Mg region


    teff = bestpars['teff']
    logg = bestpars['logg']
    feh = bestpars['fe']
    vsini = bestpars['vsini']
    psf = bestpars['psf']

    lib = smsyn.library.read_hdf(pipe.libfile, wavlim=(segs[0][0], segs[-1][-1]))
    spec = smsyn.io.spectrum.read_fits(pipe.smfile)

    fullwav = spec.wav
    # fullmod = lib.synth(lib.wav, teff, logg, feh, vsini, psf, rot_method='rotmacro')
    fullmod = lib.synth(spec.wav, teff, logg, feh, vsini, psf, rot_method='rotmacro')
    fullspec = spec.flux
    allres = spec.flux - fullmod
    wavmask = pipe.wav_exclude

    # print "Best parameters:\n", bestpars[['teff', 'logg', 'fe', 'vsini']]

    fig = pl.figure(figsize=(22, 13))
    pl.subplot2grid((3, 4), (0, 0))
    pl.subplots_adjust(bottom=0.07, left=0.03, right=0.95, wspace=0.01, hspace=0.15, top=0.95)

    pl.suptitle(title, fontsize=28, horizontalalignment='center')

    for i, seg in enumerate(segs):
        pl.subplot2grid((3, 4), (i, 0), colspan=3)

        crop = np.where((fullwav >= seg[0]) & (fullwav <= seg[1]))[0]

        pl.plot(fullwav[crop], fullspec[crop], color='0.2', linewidth=3)
        pl.plot(fullwav[crop], fullmod[crop], 'b-', linewidth=2)

        pl.axhline(0, color='r', linewidth=2)
        pl.plot(fullwav[crop], fullspec[crop] - fullmod[crop], 'k-', color='0.2', linewidth=3)

        for w0,w1 in wavmask:
            if w0 > seg[0] or w1 < seg[1]:
                pl.axvspan(w0,w1, color='LightGray')

        pl.xticks(fontsize=16)
        pl.yticks(pl.yticks()[0][1:], fontsize=16)
        pl.ylim(-0.2, 1.1)
        pl.xlim(seg[0], seg[1])

        ax = pl.gca()
        ax.tick_params(pad=5)

    pl.xlabel('Wavelength [$\\AA$]', fontsize=20)


    # ACF plot
    fullspec[~np.isfinite(fullspec)] = 1.0
    lag, acftspec = ACF(fullspec)
    lag, acfmspec = ACF(fullmod)
    dv = loglambda_wls_to_dv(fullwav, nocheck=True)
    dv = lag * dv

    pl.subplot2grid((3, 4), (2, 3))
    pl.plot(dv, acftspec, 'k.', color='0.2', linewidth=3, label='ACF of spectrum')
    pl.plot(dv, acfmspec, color='b', linewidth=2, label='ACF of model')
    locs, labels = pl.yticks()
    pl.yticks(locs, [''] * len(locs))
    pl.xticks([-50, 0, 50], fontsize=16)
    pl.xlim(-100, 100)
    crop = np.where((dv > -100) & (dv < 200))[0]
    pl.ylim(min(acfmspec[crop]), max(acfmspec[crop]) + 0.2 * max(acfmspec))
    # pl.annotate('ACF', xy=(0.15, 0.85), xycoords='axes fraction')
    ax = pl.gca()
    ax.tick_params(pad=5)

    # CCF plot
    lag, ccftspec = CCF(fullspec, fullmod)
    dv = loglambda_wls_to_dv(fullwav, nocheck=True)
    dv = lag * dv

    pl.plot(dv, ccftspec, color='0.2', linewidth=3, linestyle='dotted', label='CCF of model w/\nspectrum')
    pl.xlabel('$\\Delta v$ [km s$^{-1}$]', fontsize=20)
    ax = pl.gca()
    ax.tick_params(pad=5)

    pl.legend(numpoints=2, loc='best', fontsize=12)

    ax = pl.gca()

    try:
        mstar, rstar = bestpars['iso_mass'], bestpars['iso_radius']
        hasiso = True
    except KeyError:
        hasiso = False

    pl.subplot2grid((3, 4), (0, 3))
    afs = 22
    # plotiso()
    plotHR(os.path.join(smsyn.CAT_DIR, 'library_v12.csv'), teff, logg)
    ax = pl.gca()
    ax.set_xticks([6500, 5500, 4500, 3500])

    labels = ['teff', 'logg', 'fe', 'vsini']
    names = ['T$_{\\rm eff}$', '$\\log{g}$', 'Fe/H', '$v\\sin{i}$']
    units = ['K', '', '', 'km s$^{-1}$']

    if hasiso:
        labels.append('iso_mass')
        labels.append('iso_radius')

        names.append("M$_{\\star}$")
        names.append("R$_{\\star}$")

        units.append('M$_{\\odot}$')
        units.append('R$_{\\odot}$')

    i = 0
    for k, n, u in zip(labels, names, units):

        val = bestpars[k]
        if k+'_err' in bestpars.keys():
            err = bestpars[k+'_err']
        elif k + '_err1' in bestpars.keys():
            err = bestpars[k + '_err1']
        elif k == 'vsini':
            err = val * 0.25

        if k == 'vsini':
            if val < 2:
                pl.annotate("{} $\leq$ 2 {}".format(n, u),
                            xy=(0.76, 0.575 - 0.04 * i),
                            xycoords="figure fraction",
                            fontsize=afs)
            elif val >= 70:
                pl.annotate("{} $\geq$ 70 {}".format(n, u),
                            xy=(0.76, 0.575 - 0.04 * i),
                            xycoords="figure fraction",
                            fontsize=afs)
            else:
                err = smsyn.plotting.utils.round_sig(err, 2)
                val, err, err = smsyn.plotting.utils.sigfig(val, err)
                pl.annotate("{} = {} $\\pm$ {} {}".format(n, val, err, u),
                            xy=(0.76, 0.575 - 0.04 * i),
                            xycoords="figure fraction",
                            fontsize=afs)
        else:
            err = smsyn.plotting.utils.round_sig(err, 2)
            val, err, err = smsyn.plotting.utils.sigfig(val, err)
            # print n, val, err

            pl.annotate("{} = {} $\\pm$ {} {}".format(n, val, err, u),
                        xy=(0.76, 0.575 - 0.04 * i),
                        xycoords="figure fraction",
                        fontsize=afs)
        i += 1

    pl.savefig(outfile)

    return pl.gcf()
