"""Module that handels the fitting pipeline for hires
"""
import smsyn
import os
import numpy as np
import pandas as pd
import smsyn.io.spectrum
import smsyn.library
import smsyn.specmatch

PACKAGE_DIR = os.path.dirname(smsyn.__file__)

class Pipeline(object):
    """Pipeline object

    Top level controller for the K2 photometric pipeline.

    Args:
        smfile (str): path to output file where all intermediate results are 
            stored
        libfile (str): path to library file
        segfile (Optional[str]): path to csv file that stores the segment 
            begining and endings. If None, it's read in from smsyn.inst.hires
        wav_excludefile (Optionla[str]): path to csv file that stores
            the wavelength regions that we exclude in our fits. If
            None, it's read in from symsyn.libraries
    """
    valid_modules = ['k2phot','terra','terramulti']

    def __init__(self, smfile, libfile, segfile=None, wav_excludefile=None):
        if segfile is None:
            segfile = os.path.join(PACKAGE_DIR,'inst/hires/segments.csv')
        if wav_excludefile is None:
            wav_excludefile = os.path.join(
                PACKAGE_DIR,'models/coelho05/coelho05_wavmask.csv'
            )

        wav_exclude = pd.read_csv(wav_excludefile,comment='#')
        wav_exclude = np.array(wav_exclude)
        segments = pd.read_csv(segfile,comment='#')
        segments = np.array(segments)

        self.smfile = smfile
        self.libfile = libfile
        self.wav_exclude = wav_exclude
        self.segments = segments

    def grid_search(self, debug=False):
        spec = smsyn.io.spectrum.read_fits(self.smfile)

        # Load up the model library
        lib = smsyn.library.read_hdf(self.libfile,wavlim=[4000,4100])
        param_table = lib.model_table
        param_table['vsini'] = 5
        param_table['psf'] = 0

        # Determine spacing of coarse grid
        idx_coarse = param_table[
            param_table.teff.isin([4500,5000,5500,6000,6500,7000]) &
            param_table.logg.isin([1.0,3.0,5.0]) &
            param_table.fe.isin([-1.0,-0.5,0,0.5])
        ].index

        segments = self.segments
        if debug:
            idx_coarse = param_table[
                param_table.teff.isin([5500,6000]) &
                param_table.logg.isin([3.0,5.0]) &
                param_table.fe.isin([0.0,0.5])
            ].index
            segments = [segments[0]]
            
        # Determine spacing of fine grid
        idx_fine = param_table[~param_table.fe.isin([0.2])].index

        for segment in segments:
            param_table = smsyn.specmatch.grid_search(
                spec, self.libfile, segment, self.wav_exclude, param_table, 
                idx_coarse, idx_fine
            )

            extname = 'grid_search_%i' % segment[0]
            smsyn.io.fits.write_dataframe(self.smfile, extname,param_table,)
            
