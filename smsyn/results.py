import h5py
import numpy as np
import pandas as pd
from scipy import optimize
import lmfit
from astropy.io import fits

#from smsyn import conf
#from smsyn import smio
from smsyn import continuum
from smsyn import specmatch
#from smsyn import coelho
from smsyn import fftspecfilt
from smsyn import wlmask
from smsyn import pdplus



class SpecMatchResults(object):
    def __init__(self, grid_result, polishing_result):
        """SpecMatch Results

        Class to store or read results from a SpecMatch run.

        Args:
            grid_result (DataFrame): output from `smsyn.specmatch.grid_search`
            polishing_result (list of dicts): output from `smsyn.specmatch.polish`

        """
        
        self.grid_result = grid_result
        self.polishing_result = polishing_result

        self.bestfit = {'teff': [], 'logg': [], 'vsini': [], 'fe': [], 'psf': []}
        ikeys = self.bestfit.keys()
        for seg in polishing_result:
            result = seg['result']
            params = result.params
            for k in ikeys:
                self.bestfit[k].append(params[k].value)
                self.bestfit[k+'_vary'] = params[k].vary
                self.bestfit[k+'_min'] = params[k].min
                self.bestfit[k+'_max'] = params[k].max

        for k in ikeys:
            self.bestfit['u'+k] = np.std(self.bestfit[k])
            self.bestfit[k] = np.mean(self.bestfit[k])

        for k in self.bestfit.keys():
            if not np.isfinite(self.bestfit[k]):
                self.bestfit[k] = -999
                
    def to_fits(self, outfile, clobber=True):
        """Save to FITS

        Save a SpecMatchResults object as a mutli-extension fits file.

        Args:
            outfile (string): name of output file name
            clobber (bool): if true, will overwrite existing file
            
        """

        # Save the grid search results
        columns = []
        for i,col in enumerate(self.grid_result.columns):
            colinfo = col
            coldata = self.grid_result[col].values
            fitscol = fits.Column(array=coldata, format='D', name=col)

            columns.append(fitscol)

        grid_hdu = fits.BinTableHDU.from_columns(columns)

        polish_hdus = []
        # Save the polishing results
        for i,seg in enumerate(self.polishing_result):
            columns = []
            header = fits.Header()
            for k in seg.keys():
                colinfo = k
                coldata = seg[k]
                if k == 'result':
                    mini = seg[k]
                    for p in mini.params.keys():
                        header[p] = mini.params[p].value
                else:
                    fitscol = fits.Column(array=coldata, format='D', name=k)

                columns.append(fitscol)
                
            polish_hdus.append(fits.BinTableHDU.from_columns(columns, header=header))

        
        fitsheader = fits.Header()

        ext_defs = {'EXT0': 'PrimaryHDU',
                    'EXT1': 'Grid search results'}
        for i,seg in enumerate(self.polishing_result):
            ext_defs['EXT%d' % (i+2)] = 'Polishing results for wav0=%d' % seg['wav'].min()

        fitsheader.update(ext_defs)
        fitsheader.update(self.bestfit)
        
        primary_hdu = fits.PrimaryHDU(header=fitsheader)
        hdu_list = fits.HDUList([primary_hdu, grid_hdu]+polish_hdus)

        hdu_list.writeto(outfile, clobber=clobber)




def getpars_lincomb(h5file,group,usemask=True,plot=False,outtext=False):
    """
    Get Parameters: Linear Combintations

    Find the best fitting spectrum by taking linear combinations of
    the best 8 matches.
    
    Parameters
    ----------
    h5file : path to h5 file (matches must be computed)
    group : uses:
            group['arr'] spectra, sorted by fchi
            group['smres'] results, sorted by fchi

    Returns
    -------
    par : dictionary with stellar parameters
          - teff
          - logg
          - fe
          - vsini
    """
    ntop = 8 # construct matches from the top 8 matches
    with h5py.File(h5file,'r+') as h5:
        smres = h5['%i/smres' % group][:]
        lspec = h5['%i/lspec' % group][:]
        tspec = h5['%i/tspec' % group][:]
        
        # Figure out the error
        ord = conf.segdf.ix[group,'ord']
        serr = np.median(smio.getspec_fits(h5.attrs['obs'])[ord]['serr'])

    smres = pd.DataFrame(smres)
    smres = smres_add_chi(smres)

    ### > new
    assert np.allclose(smres.fchi,smres.sort_values(by='fchi').fchi),\
        "error not sorted by fchi"
    
    smresbest = smres.iloc[:ntop]    
    libbest = lspec[:ntop]['lspec']

    ### > old
    # smresbest = smres.sort('chi').iloc[:8]
    # libbest = arr[smresbest.index]['libspec']

    tspec = pd.DataFrame(tspec)
    tspec = tspec.rename(columns={'tspec':'s'}).to_records(index=False)
    tspec = np.array(tspec)
    w = tspec['w']

    mspec = tspec.copy()
    mask = np.zeros(w.size).astype(bool)
    
    if usemask:
        maskpath='/Users/petigura/Research/SpecMatch/config/wav_mask.csv'
        dfmask = wlmask.loadmask(maskpath=maskpath)
        mask = wlmask.getmask(w,dfmask,mode='exclude')

        #print "Group %i masking out %i pixels" % (group, mask.sum())
        
    # First cut synthesize model spectrum
    p1,chi = optimize.nnls(libbest[:,~mask].T,tspec['s'][~mask].T)
    p1 = p1 / np.sum(p1)    
    mspec['s'] = np.dot(p1,libbest)
    
    # Renormalize the target spectrum to the continuum level
    # Fit continuum of target
    # Fit continuum of model
    # target / (continuum target) * (continuum model)
    c = continuum.cfit(tspec)    
    mspec2 = mspec.copy()
    mspec2['s'] += np.random.randn(mspec2.size) * serr 
    mc = continuum.cfit(mspec2)
    tspec['s'] = tspec['s'] / c * mc
    p1,chi = optimize.nnls(libbest[:,~mask].T,tspec['s'][~mask].T)
    p1 = p1 / np.sum(p1)    
    mspec['s'] = np.dot(p1,libbest)

    keys = 'teff logg fe vsini'.split()
    par = dict( [(k,np.dot(smresbest[k],p1)) for k in keys]) 
    par['obs'] = smres.iloc[0]['targobs']

    if plot:
        from matplotlib.pylab import plt
        import smplots

        res = tspec['s']-mspec['s']
        res = ma.masked_array(res)

        fig,axL = plt.subplots(nrows=3,figsize=(8,5),sharey=True)
        fig.set_tight_layout(True)

        @smplots.stackax(axL)
        def diag_plot():
            plt.plot(w,tspec['s'])
            plt.plot(w,mspec['s'])
            plt.plot(w,res)
            smplots.axvspan_mask(w,mask)
        diag_plot()
        plt.ylim(-0.2,1.2)
        plt.xlabel('Wavelength (A)')
        plt.ylabel('Intensity')

    if outtext:
        pd.set_option('precision',3) 
        smresbest['p'] = p1
        s = smresbest['teff logg fe vsini chi p'.split()].to_string()
        s +="\n\n%.2e" % ( np.sum(res**2)/res.count() ) 
        s +="\n\n%(teff)i %(logg).2f %(fe).2f %(vsini).2f" % par

        ax = fig.get_axes()[0]
        plt.sca(ax)
        ax.text(1.01,1,s,transform=ax.transAxes,ha='left',va='top',
                family='monospace')
        plt.title(h5file+' %i' % group )
        plt.draw()
        fig.set_tight_layout(False)
        fig.subplots_adjust(right=0.7,top=0.95)
        plt.draw()

    return par    

def smres_add_chi(smres):
    """
    SpecMatch Results Chi

    Convert the fields smres in to a single figure of merit.
    """
    smres['chi'] = smres['fchi']
    smres['chi'] = smres['chi'] - min(smres['chi']) + 1
    return smres

def getpars_polish(targd,par0,h5=None,usemask=True,snr=None,ret0=False):
    """
    Read in an h5 structure and extract the best fit parameters.

    2. Fit all the orders with a synthesized coelho model.

    Parameters
    ----------
    targd : dictionary with `name`, `obs`, and `type` keywords
    par0 : dictionary with best guess teff, logg, fe, vsini
    h5 : h5 object, to save out results. If None, results are just returned
    """

    # List of wavelength regions to use.
    wloL = conf.wloL_fm

    nwlo = len(wloL)
    segdf = conf.segdf.ix[wloL].copy()

    # Deal with orders using lists of dictionaries
    def get_tspec(wlo):
        """
        Load up the target spectrum and wavelength mask
        """
        segd = segdf.ix[wlo]
        targsegd = dict(targd,**segd)
        tspec = smio.getspec_h5(**targsegd)
        if snr!=None:
            tspec = noise_up_spec(tspec,snr)

        w = tspec['w']
        s = tspec['s']
        serr = tspec['serr']
        c = continuum.cfit(tspec);
        return dict(w=w,s=s,serr=serr,c=c)

    tspecL = map(get_tspec,wloL)

    def model_list(params):
        """
        Wrapper around coelho_synth that produces a model at each wavelength
        region
        """

        teff = params['teff'].value
        logg = params['logg'].value
        fe = params['fe'].value
        vsini = params['vsini'].value

        def get_mspec(wlo,tspec):
            segd = segdf.ix[wlo]
            mspec = coelho.coelho_synth(teff,logg,fe,vsini,par0['psfsig'],
                                        **segd)


#            mc = continuum.cfit(mspec) # model continuum 
#            tspec_cfit = tspec['s']*mc/tspec['c'] # normalized target spectrum
#            tspec_cfit /= params['cscale_%i' % wlo]
#
            mspec = mspec['s']
            res = tspec['s'] - mspec
            fres = fftspecfilt.fftbandfilt(res,whi=300)
            return dict(res=res,fres=fres,mspec=mspec)
        
        mspecL = map(get_mspec,wloL,tspecL)
        return mspecL


    # Intialize parameters
    params = lmfit.Parameters()
    keys = 'teff logg fe vsini'.split()
    for k in keys:
        params.add(k,value=par0[k])
#    for wlo in wloL:
#        params.add('cscale_%i' % wlo,value=1.0)
#
    params['vsini'].min=0



    # List of model spectra with 
    mspecL0 = model_list(params)
    
    def get_mask(tspec,mspec):
        mspec = mspec['mspec']
        # Add in mask if it exists
        mask = np.zeros(tspec['s'].size).astype(bool)
        if usemask:
            maskpath='/Users/petigura/Research/SpecMatch/config/wav_mask.csv'
            dfmask = wlmask.loadmask(maskpath=maskpath)
            dfmask_clip = wlmask.specmask(tspec) # outlier rejection

            # Ignore index prevents multiple regions from having the same index
            dfmask = pd.concat([dfmask,dfmask_clip],ignore_index=True)
            mask = wlmask.getmask(tspec['w'],dfmask,mode='exclude')
            
        return mask
    
    maskL = map(get_mask,tspecL,mspecL0)
        
    def residuals(params):
        mspecL = model_list(params)
        res = []
        for tspec,mspec,mask in zip(tspecL,mspecL,maskL):
            res += [ (mspec['fres'] / tspec['serr'])[~mask] ]
        res = np.hstack(res)
        return res

    if ret0:
        return residuals(params)

    # If ftol is 1e-2, carver and my laptop give different answers ~80 K level.
    out = lmfit.minimize(residuals,params,ftol=1e-8)
    mspecL = model_list(out.params)

    outpars = {}
    for k in params.keys():
        outpars[k] = params[k].value
    res = residuals(out.params)
    outpars['chi'] = np.sum(res**2/res.size)

    if h5!=None:
        for tspec,mspec,mask,wlo in zip(tspecL,mspecL,maskL,wloL):
            gname = "polish_%i" % wlo
            g = h5.create_group(gname)
            d = dict(tspec,**mspec) 
            d['mask'] = mask
            df = pd.DataFrame(d)
            df = pdplus.df_to_ndarray(df)
            h5.create_dataset('%s/res' % gname,data=df)

    return outpars


def SMpars(smres,ntop=10,ver=False):
    """
    SpecMatch Parameters

    Convert a chi2 array into best fit parameters
    
    Parameters
    ----------
    smres : DataFrame with the following keys
            teff logg fe

    Returns
    -------
    dres : dictionary of best fit parameters
    """

    smres = smres_add_chi(smres)
    smresbest = smres.sort_values(by='chi').iloc[:ntop]

    kwmean = 'teff logg fe'.split()
    kwmean_lib = 'teff logg fe'.split()
    chauvpass = np.zeros(len(smresbest))
    for k in kwmean_lib:
        chauvpass += chauvent(smresbest[k]).astype(int)

    dres = {}
    chauvpass = ( chauvpass == len(kwmean_lib) )
#    if not chauvpass.all():
#        if ver:
#            print "%i matches are inliers" % chauvpass.sum()
#        dfbest = dfbest.iloc[chauvpass]
#
    for k,kl in zip(kwmean,kwmean_lib):
        dres[k] = wmean(smresbest[kl],smresbest['chi'])    
        dres['u'+k] = wmean(smresbest[kl],smresbest['chi'],stdev=True)

    return dres

from scipy.special import erfinv

def chauvent(residuals):
    """
    Chauvenet's criterion
     
    PURPOSE:
    Given the residuals of a fit, this routine returns the indices
    of points that pass Chauvenet's criterion.  You can also return
    the number of points that passed, the indices of the points
    that failed, the number of points that failed, and a byte mask
    where 1B represents a passing point and 0B represents a failing
    point.
     
    EXPLANATION:
    Chauvenet's criterion states that a datum should be discarded if 
    less than half an event is expected to be further from the mean of 
    the data set than the suspect datum.  You should only use this
    criterion if the data are normally distributed.  Many authorities
    believe Chauvenet's criterion should never be applied a second
    time.  Not me.
     
    Parameters
    ----------
    residuals : The residuals of a fit to data.
     
    Returns
    -------
    mask : binary mask of elements that pass criterion 

    History
    -------
    Written by Tim Robishaw, Berkeley  Dec 13, 2001
    Modified for python Erik Petigura Dec 30, 2013
    """

    assert (np.isfinite(residuals)==True).all(),"residuals must be final"
    N = residuals.size 
    assert N > 1,"Chauvenet's criterion requires N > 1!"
    
    mean = np.mean(residuals)
    rms  = np.sqrt( np.sum((residuals-mean)**2)/ (N-1) )

    # FIND INVERSE ERROR FUNCTION OF (1 - 0.5/N)...
    # MAKE A MASK OF POINTS THAT PASSED CHAUVENET'S CRITERION...
    mask = np.abs(residuals - mean) <  ( 1.4142 * rms * erfinv(1.- 0.5/N) )
    return mask

def wmean(x,chi2,stdev=False):
    """
    Return the mean of x weighted by 1/chi2
    """
    w = 1. / chi2
    if stdev:
        deltax = (x-np.mean(x))**2
        sd = np.sqrt(np.sum(w*deltax)/np.sum(w))

    return np.sum(x*w)/np.sum(w)
