"""Module to define the Library class

This module defines the Library object used for specmatch-synth


"""
import numpy as np
import pandas as pd
import h5py

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

        wavelength (array): 1-d vector containng the wavelength scale
            for the model spectra

        model_spectra (array): array containing all model spectra
            ordered so that the they can be referenced by the indicies
            contained in the `model_table`.
    """
    header_required_keys = ['model_name', 'model_reference']
    target_chunk_bytes = 100e3 # Target number of bytes are per hdf chunk

    def __init__(self, header, model_table, wavelength, model_spectra, 
                 wavlim=None):
        for key in self.header_required_keys:
            assert key in header.keys(), "{} required in header".format(key)

        self.header = header
        self.model_table = model_table
        self.wavelength = wavelength
        self.model_spectra = model_spectra
        self.wavlim = wavlim

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
            h5['wavelength'] = self.wavelength

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

    def synth(self, wav, teff, logg, fe, vsini, psf, interp_mode='trilinear'):
        """Synthesize a model spectrum

        Interpolate between points in the model grid and synthesize a
        spectral region for a given set of stellar parameters.

        """
        pass
    
def read_hdf(filename, wavlim=None):
    """Read model library grid

    Read in a model library grid from an h5 file and initialze a Library object.

    Args:
        filename (string): path to h5 file that contains the grid
            of stellar atmosphere models

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

    lib = Library(header, model_table, wavelength, model_spectra, wavlim=wavlim)
    return lib
