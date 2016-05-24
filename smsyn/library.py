"""Module to define the Library class

This module defines the Library object used for specmatch-synth


"""

import numpy as np
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
    def __init__(self, header, model_table, wavelength, model_spectra):
        for key in header_required_keys:
            assert key is in header.keys(), "{} required in header".format(key)

        self.header = header
        self.model_table
        self.wavelength = wavelength
        self.model_spectra = model_spectra

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
            h5['model_spectra'] = self.model_spectra
            h5['wavelength'] = self.wavelength
        
    def from_hdf(self, filename):
        """Read model library grid

        Read in a model library grid from an h5 file.

        Args:
            filename (string): path to h5 file that contains the grid
                of stellar atmosphere models
        
        """

        
        
        pass

    def synth(self, wav, teff, logg, fe, vsini, psf, interp_mode='trilinear'):
        """Synthesize a model spectrum

        Interpolate between points in the model grid
               and synthesize a spectral region for a given
               set of stellar parameters.
        
        pass

    
