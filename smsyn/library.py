"""Module to define the Library class

This module defines the Library object used for specmatch-synth


"""
import itertools

import numpy as np
import scipy.ndimage as nd
import scipy.interpolate
import pandas as pd
import h5py

import smsyn.restwav
import smsyn.kernels

class Library(object):
    """The Library object

    This object handles reading the model grid and associating the models with
    stellar parameters

    Args:
        header (dict): a dictionary containing metadata that describes
            the model library.  'model_name' and 'model_reference' are
            the only required keys.  (e.g. {'model_name': 'coelho05',
            'model_reference': 'Coelho et al. (2005)'})

        model_table (DataFrame): Pandas DataFrame with the following
            columns: teff, logg, fe, model_index. The model_index
            column should give the index to the model spectrum in the
            `model_spectra` array that is associated with the given
            parameters.

        wav (array): 1-d vector containng the wavelength scale
            for the model spectra

        model_spectra (array): array containing all model spectra
            ordered so that the they can be referenced by the indicies
            contained in the `model_table`.

        wavlim (2-element iterable): (optional) list, tuple, or other 2-element
            itarable that contains the upper and lower wavelengths limits to be read
            into memory
    """
    header_required_keys = ['model_name', 'model_reference']
    target_chunk_bytes = 100e3 # Target number of bytes are per hdf chunk

    def __init__(self, header, model_table, wav, model_spectra, 
                 wavlim=None):
        for key in self.header_required_keys:
            assert key in header.keys(), "{} required in header".format(key)

        self.header = header
        self.model_table = model_table
        self.wav = wav
        self.model_spectra = model_spectra
        self.wavlim = wavlim

    def __repr__(self):
        return "<smsyn.library.Library object for the {0} model library ({1})>".format(
            self.header['model_name'], self.header['model_reference'])
        
    def to_hdf(self, outfile):
        """Save model library

        Save a model library as an h5 file

        Args:
            outfile (string): path to output h5 file
        """

        with h5py.File(outfile,'w') as h5:
            for key in self.header.keys():
                h5.attrs[key] = self.header[key]

            model_table = np.array(self.model_table.to_records(index=False))
            h5['model_table'] = model_table
            h5['wav'] = self.wavelength

            # Compute chunk size for compressed library
            chunk_row = self.model_spectra.shape[0]
            chunk_col = self.target_chunk_bytes / self.model_spectra[:,0].nbytes
            chunk_col = int(chunk_col)
            chunks = (chunk_row, chunk_col)

            print "storing model spectra with chunks of size {}".format(chunks)
            dset = h5.create_dataset(
                'model_spectra', data=self.model_spectra, compression='gzip', 
                compression_opts=1, shuffle=True, chunks=chunks
            )

    def select_model(self, pars):
        """Select a model spectrum

        Grab a model spectrum from the library that corresponds
        to a given set of stellar parameters.

        Args:
            pars (3-element iterable): A 3-element tuple containing (teff, logg, and fe)
            
        Returns:
            array: model spectrum flux resampled at the new wavelengths
        
        """

        assert (pars[0] in self.model_table['teff'].values) & \
               (pars[1] in self.model_table['logg'].values) & \
               (pars[2] in self.model_table['fe'].values), \
               "The given set of parameters does not match a row in the model_table lookup table: {}".format(pars)        
        
        row = self.model_table[(self.model_table['teff'] == pars[0]) &
                                    (self.model_table['logg'] == pars[1]) &
                                    (self.model_table['fe'] == pars[2])]
        idx = row['model_index']
        spec = self.model_spectra[idx]
        
        return spec

            
    def synth(self, wav, teff, logg, fe, vsini, psf, interp_mode='trilinear'):
        """Synthesize a model spectrum

        For a given set of wavelengths teff, logg, fe, vsini, psf, compute a model spectrum by:

            1. Determine the 8 coelho models surounding the (teff,logg,fe)
            2. Perform trilinear interpolation
            3. Resample onto new wavelength scale
            4. Broaden with rot-macro turbulence
            5. Broaden with PSF (assume gaussian)

        Args:
            wav   (array): wavelengths where the model will be calculated
            teff  (float): effective temp (K)
            logg  (float): surface gravity (logg)
            fe    (float): metalicity [Fe/H] (dex)
            vsini (float): rotational velocity (km/s)
            psf   (float): sigma for instrumental profile (pixels)

        Returns:
            array: synthesized model calculated at the wavelengths specified
                in the wav argument

        """

        
        self.model_table['dteff'] = np.abs(self.model_table['teff']-teff)
        self.model_table['dlogg'] = np.abs(self.model_table['logg']-logg)
        self.model_table['dfe'] = np.abs(self.model_table['fe']-fe)

        teff1,teff2 = self.model_table.sort_values(by='dteff')['teff'].drop_duplicates()[:2]
        logg1,logg2 = self.model_table.sort_values(by='dlogg')['logg'].drop_duplicates()[:2]
        fe1,fe2 = self.model_table.sort_values(by='dfe')['fe'].drop_duplicates()[:2]

        corners = itertools.product([teff1,teff2],[logg1,logg2],[fe1,fe2])
        
        c = np.vstack( map(self.select_model,corners) )
    
        v0 = [teff1, logg1, fe1]
        v1 = [teff2, logg2, fe2]
        vi = [teff, logg, fe]

        # Perform interpolation
        if interp_mode == 'trilinear':
            s = trilinear_interp(c,v0,v1,vi)
        else:
            raise NameError, "Interpolation mode {} not implemented.".format(interp_mode)

        # Resample at the requested wavelengths
        s = scipy.interpolate.InterpolatedUnivariateSpline(self.wav, s)(wav)
            
        # Broaden with rotational-macroturbulent broadening profile
        dvel = smsyn.restwav.wav_to_dvel(wav)
        dvel0 = dvel[0]
        if np.allclose(dvel,dvel[0],rtol=1e-3,atol=1) is False:
            print "wav not uniform in loglambda, using mean dvel"
            dvel0 = np.mean(dvel)

        n = 151 # Correct for VsinI up to ~50 km/s

        # Valenti and Fischer macroturb reln ERROR IN PAPER!!!
        xi = 3.98 + (teff-5770)/650
        if xi < 0: 
            xi = 0 
    
        varr,M = smsyn.kernels.rotmacro(n,dvel0,xi,vsini)
        s = nd.convolve1d(s,M) 

        # Broaden with PSF (assume gaussian) (km/s)
        if psf > 0: s = nd.gaussian_filter(s,psf)

        return s

    
def read_hdf(filename, wavlim=None):
    """Read model library grid

    Read in a model library grid from an h5 file and initialze a Library object.

    Args:
        filename (string): path to h5 file that contains the grid
            of stellar atmosphere models
        wavlim (2-element iterable): upper and lower wavelength limits
            (in Angstroms) to load into RAM
        
    Returns:
        Library object
        
    """
    
    with h5py.File(filename,'r') as h5:
        header = dict(h5.attrs)
        model_table = pd.DataFrame.from_records(h5['model_table'][:])
        wavelength = h5['wavelength'][:]
        
        if wavlim is None:
            model_spectra = h5['model_spectra'][:]
        else:
            idxwav, = np.where(
                (wavelength > wavlim[0]) &
                (wavelength < wavlim[1])
            )
            idxmin = idxwav[0]
            idxmax = idxwav[-1] + 1 # add 1 to include last index when slicing
            model_spectra = h5['model_spectra'][:,idxmin:idxmax]
            wavelength = wavelength[idxmin:idxmax]

    lib = Library(
        header, model_table, wavelength, model_spectra, wavlim=wavlim
    )
    return lib


def trilinear_interp(c,v0,v1,vi):
    """Trilinear interpolation

    Perform trilinear interpolation as described here.
    http://en.wikipedia.org/wiki/Trilinear_interpolation

    Args:
        c (8 x n array): where C each row of C corresponds to the value at one corner
        v0 (length 3 array): with the origin
        v1 (length 3 array): with coordinates on the diagonal
        vi (length 3 array): specifying the interpolated coordinates

    Returns:
        interpolated value of c at vi
        
    """

    v0 = np.array(v0) 
    v1 = np.array(v1) 
    vi = np.array(vi) 

    vd = (vi-v0)/(v1-v0) # fractional distance between grid points

    cx0 = c[:4] # function at x0
    cx1 = c[4:] # function at x1

    cix = cx0 * (1-vd[0]) +  cx1 * vd[0]
    cixy = cix[:2] * (1-vd[1]) +  cix[2:] * vd[1]
    cixyz = cixy[0] * (1-vd[2]) +  cixy[1] * vd[2]
    return cixyz

    

