"""Module to define the Library class

This module defines the Library object used for specmatch-synth


"""
import itertools

import numpy as np
import scipy.ndimage as nd
import scipy.interpolate
import pandas as pd
import h5py
from scipy.spatial import Delaunay

import smsyn.kernels
import smsyn.wavsol

# assert the two floats are equal if they are closer than this amount
FLOAT_TOL = 1e-3 

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

        wav (array): 1-d vector containng the wavelength scale for the
            model spectra

        model_spectra (array): array containing all model spectra
            ordered so that the they can be referenced by the indicies
            contained in the `model_table`.

        wavlim (2-element iterable): (optional) list, tuple, or other
            2-element itarable that contains the upper and lower
            wavelengths limits to be read into memory

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
        _outstr = "<smsyn.library.Library {0} model library ({1})>".format(
            self.header['model_name'], self.header['model_reference']
        )
        return _outstr
        
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
            h5['wav'] = self.wav

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
            pars (3-element iterable): A 3-element tuple containing teff, logg,
                and fe
            
        Returns:
            array: model spectrum flux resampled at the new wavelengths
        
        """
        arr = np.array(self.model_table['teff logg fe'.split()])
        arr-=pars
        idx = np.where(np.sum(np.abs(arr) < 1e-3,axis=1)==3)[0]
        assert len(idx)==1, "model at {} not found".format(pars)
        spec = self.model_spectra[idx]
        return spec

    def _trilinear_interp(self, teff, logg, fe):
        self.model_table['dteff'] = np.abs(self.model_table['teff'] - teff)
        self.model_table['dlogg'] = np.abs(self.model_table['logg'] - logg)
        self.model_table['dfe'] = np.abs(self.model_table['fe'] - fe)

        teff1,teff2 = self.model_table.sort_values(by='dteff')['teff'] \
                                      .drop_duplicates()[:2]
        logg1,logg2 = self.model_table.sort_values(by='dlogg')['logg'] \
                                      .drop_duplicates()[:2]
        fe1,fe2 = self.model_table.sort_values(by='dfe')['fe'] \
                                  .drop_duplicates()[:2]

        corners = itertools.product([teff1,teff2],[logg1,logg2],[fe1,fe2])
        c = np.vstack( map(self.select_model,corners) )
        v0 = [teff1, logg1, fe1]
        v1 = [teff2, logg2, fe2]
        vi = [teff, logg, fe]
        flux = trilinear_interp(c,v0,v1,vi)
        return flux

    def _simplex_interp(self, teff, logg, fe, model_indecies):
        """
        Perform barycentric interpolation on simplecies

        Args:
            teff (float): effective temperature
            logg (float): surface gravity
            fe (float): metalicity
            model_indecies (array): models to use in the interpolation

        Returns:
            array: spectrum at teff, logg, fe

        """

        ndim = 3
        model_table = np.array(self.model_table['teff logg fe'.split()])
        points = model_table[model_indecies]
        tri = Delaunay(points) # Delaunay triangulation
        p = np.array([teff, logg, fe]) # cartesian coordinates

        simplex = tri.find_simplex(p)
        r = tri.transform[simplex,ndim,:]
        Tinv = tri.transform[simplex,:ndim,:ndim]
        c = Tinv.dot(p - r)
        c = np.hstack([c,1.0-c.sum()]) # barycentric coordinates

        # (3,many array)
        model_indecies_interp = model_indecies[tri.simplices[simplex]]
        model_spectra = self.model_spectra[model_indecies_interp,:]
        flux = np.dot(c,model_spectra)
        if simplex==-1:
            return np.ones_like(flux)
        return flux

    def interp_model(self, teff, logg, fe, **interp_kw):
        interp_mode = interp_kw['mode']
        arr = np.array(self.model_table['teff logg fe'.split()])
        arr_params = np.array([teff, logg, fe])
        arr-=arr_params
        idx = np.where(np.sum(np.abs(arr) < FLOAT_TOL,axis=1)==3)[0]
        if len(idx)==1:
            flux = self.model_spectra[idx][0]
            return flux
        
        if interp_mode == 'trilinear':
            flux = self._trilinear_interp(teff, logg, fe)
        elif interp_mode == 'simplex':
            model_indecies = interp_kw['model_indecies']
            flux = self._simplex_interp(teff, logg, fe, model_indecies)
        else:
            errmsg = "Interpolation mode {} not implemented.".format(
                interp_mode
            )
            raise NameError, errmsg
        return flux

    def _broaden_rotmacro(self, flux, dvel, teff, vsini):
        n = 151 # Correct for VsinI up to ~50 km/s
        xi = 3.98 + (teff - 5770.0) / 650.0
        if xi < 0: 
            xi = 0 
    
        varr, M = smsyn.kernels.rotmacro(n, dvel, xi, vsini)
        flux = nd.convolve1d(flux, M) 
        return flux

    def _broaden_rot(self, flux, dvel, vsini):
        n = 151 # Correct for VsinI up to ~50 km/s
        varr, M = smsyn.kernels.rot(n, dvel, vsini)
        flux = nd.convolve1d(flux, M) 
        return flux

    def _broaden(self, wav, flux, psf=None, rot_method='rotmacro', teff=None, 
                 vsini=None):
        """
        Args:
            wav (array): wavelength
            flux (array): fluxes
            psf (float): width of gaussian psf 
            rot_method (str): Treatment of rotation. If 'rotmacro', then teff and
               vsini must be set. If 'none', then no rotation rotational
               broadening is used.
        """
        # Broaden with rotational-macroturbulent broadening profile
        dvel = smsyn.wavsol.wav_to_dvel(wav)
        dvel0 = dvel[0]
        if np.allclose(dvel, dvel[0], rtol=1e-3, atol=1) is False:
            print "wav not uniform in loglambda, using mean dvel"
            dvel0 = np.mean(dvel)

        if rot_method=='rotmacro':
            flux = self._broaden_rotmacro(flux, dvel0, teff, vsini)
        elif rot_method=='rot':        
            flux = self._broaden_rot(flux, dvel0, vsini)
        else:        
            assert False,'invalid rot_method' 

        if psf is not None:
            # Broaden with PSF (assume gaussian) (km/s)
            if psf > 0: 
                flux = nd.gaussian_filter(flux,psf)

        return flux
            
    def synth(self, wav, teff, logg, fe, vsini, psf, rot_method,
              interp_kw=None):
        """Synthesize a model spectrum

        For a given set of wavelengths teff, logg, fe, vsini, psf,
        compute a model spectrum by:

            1. Determine the 8 coelho models surounding the (teff,logg,fe)
            2. Perform trilinear interpolation
            3. Resample onto new wavelength scale
            4. Broaden with rotational kernel {'rot','rotmacro'}
            5. Broaden with PSF (assume gaussian)

        Args:
            wav (array): wavelengths where the model will be calculated
            teff (float): effective temp (K)
            logg (float): surface gravity (logg)
            fe (float): metalicity [Fe/H] (dex)
            vsini (float): rotational velocity (km/s)
            psf (float): sigma for instrumental profile (pixels)

        Returns:
            array: synthesized model calculated at the wavelengths specified
                in the wav argument
        """
        if interp_kw is None:
            interp_kw = dict(mode='trilinear')
            
        flux = self.interp_model(teff, logg, fe, **interp_kw)
        flux = np.interp(wav, self.wav, flux) # Resample at input wavelengths
        flux = self._broaden(
            wav, flux, psf=psf, rot_method=rot_method, teff=teff, vsini=vsini
        )
        return flux

    def synth_lincomb(self, wav, model_indecies, coeff, vsini, psf):
        assert (coeff >= 0).all(), "coeff must be postive"
        coeff /= coeff.sum()
        flux = np.dot(coeff, self.model_spectra[model_indecies])
        flux = np.interp(wav, self.wav, flux) # Resample at input wavelengths
        flux = self._broaden(
            wav, flux, psf=psf, rot_method='rotmacro', teff=5700, vsini=vsini
        )
        return flux
    
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
        wav = h5['wav'][:]
        
        if wavlim is None:
            model_spectra = h5['model_spectra'][:]
        else:
            idxwav, = np.where( (wav > wavlim[0]) & (wav < wavlim[1]))
            idxmin = idxwav[0]
            idxmax = idxwav[-1] + 1 # add 1 to include last index when slicing
            model_spectra = h5['model_spectra'][:,idxmin:idxmax]
            wav = wav[idxmin:idxmax]

    lib = Library(header, model_table, wav, model_spectra, wavlim=wavlim)
    return lib


def trilinear_interp(c,v0,v1,vi):
    """Trilinear interpolation

    Perform trilinear interpolation as described here.
    http://en.wikipedia.org/wiki/Trilinear_interpolation

    Args:
        c (8 x n array): where C each row of C corresponds to the value at one 
            corner
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
