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

PARAM_KEYS = ['teff', 'logg', 'vsini', 'fe', 'psf']

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

        self.bestfit = {k: [] for k in PARAM_KEYS}
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
            self.bestfit['u'+k] = np.std(self.bestfit[k]) / np.sqrt(len(polishing_result))
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
                elif k == 'objective':
                    header['OBJFUNC'] = coldata
                else:
                    fitscol = fits.Column(array=coldata, format='D', name=k)
                    columns.append(fitscol)


            polish_hdus.append(fits.BinTableHDU.from_columns(columns, header=header))
                
            
            

        
        fitsheader = fits.Header()

        # Add descriptions of extensions into primary header
        ext_defs = {'EXT0': 'PrimaryHDU',
                    'EXT1': 'Grid search results'}
        for i,seg in enumerate(self.polishing_result):
            ext_defs['EXT%d' % (i+2)] = 'Polishing results for wav0=%d' % seg['wav'].min()

        fitsheader.update(ext_defs)
        fitsheader.update(self.bestfit)
        
        primary_hdu = fits.PrimaryHDU(header=fitsheader)
        hdu_list = fits.HDUList([primary_hdu, grid_hdu]+polish_hdus)

        hdu_list.writeto(outfile, clobber=clobber)

def read_fits(filename):
    """Read results from fits file

    Read in a results object as saved by the results.SpecMatchResults.to_fits
    method into a new SpecMatchResults object

    Args:
        filename (string): path to fits file
        
    Returns:
        SpecMatchResults object
        
    """

    class _Store_Params(object):
        def __init__(self, header):
            params = lmfit.Parameters()
            for k in PARAM_KEYS:
                hkey = k.upper()
                val = header[hkey]
                min_limit = header[hkey+'_MIN']
                max_limit = header[hkey+'_MAX']
                var = header[hkey+'_VARY']
                
                if min_limit == -999:
                    min_limit = -np.inf
                if max_limit == -999:
                    max_limit = np.inf
                
                params.add(k, value=val,
                           min=min_limit,
                           max=max_limit,
                           vary=var
                           )

            self.params = params
            
                
    hdulist = fits.open(filename)
    header = hdulist[0].header

    grid_results = pd.DataFrame(hdulist[1].data)
    
    output = []
    for hdu in hdulist[2:]:

        head = hdu.header
        df = pd.DataFrame(hdu.data)

        fake_result_obj = _Store_Params(header)

        d = dict(
            model=df['model'].values, 
            continuum=df['continuum'].values, 
            wav=df['wav'].values,
            resid=df['resid'].values, 
            objective=head['OBJFUNC'],
            result=fake_result_obj
            )

        output.append(d)


    smresults = SpecMatchResults(grid_results, output)
                
    return smresults
    


