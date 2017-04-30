"""Module that handels the fitting pipeline for hires
"""
import os
import cPickle as pickle

import numpy as np
import pandas as pd
import lmfit

import smsyn
import smsyn.io.spectrum
import smsyn.library
import smsyn.specmatch

PACKAGE_DIR = os.path.dirname(smsyn.__file__)

class Pipeline(object):
    """Pipeline object

    Generic container object that carries the data products associate
    with the keck runs

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
        self.pklfn = smfile.replace('.fits','.pkl')

        self.libfile = libfile
        self.wav_exclude = wav_exclude
        self.segments = segments


    def _get_spec_segment(self, segment):
        spec = smsyn.io.spectrum.read_fits(self.smfile)
        spec = spec[(segment[0] < spec.wav) & (spec.wav < segment[1])]
        return spec

    def to_pickle(self):
        with open(self.pklfn,'w') as f:
            pickle.dump(self,f)

def grid_search(pipe, debug=False):
    """
    Run grid search
    
    Args:
        pipe (Pipeline): object

    """

    # Load up the model library
    lib = smsyn.library.read_hdf(pipe.libfile, wavlim=[4000,4100])
    param_table = lib.model_table
    param_table['vsini'] = 5
    param_table['psf'] = 0

    # Determine spacing of coarse grid
    idx_coarse = param_table[
        param_table.teff.isin([4500,5000,5500,6000,6500,7000]) &
        param_table.logg.isin([1.0,3.0,5.0]) &
        param_table.fe.isin([-1.0,-0.5,0,0.5])
    ].index

    segments = pipe.segments
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
        spec = pipe._get_spec_segment(segment)
        print "Grid search: {}".format(spec.__repr__())

        param_table = smsyn.specmatch.grid_search(
            spec, pipe.libfile, pipe.wav_exclude, param_table, 
            idx_coarse, idx_fine
        )

        extname = 'grid_search_%i' % segment[0]
        smsyn.io.fits.write_dataframe(pipe.smfile, extname, param_table,)

def lincomb(pipe):
    """
    Figure out optimal parameters for individual segments
    
    Args:
        pipe (Pipeline): object

    """
    ntop = 8
    print "teff logg fe vsini psf rchisq0 rchisq1"
    for segment in pipe.segments:
        extname = 'grid_search_%i' % segment[0]
        param_table = smsyn.io.fits.read_dataframe(pipe.smfile, extname)
        spec = pipe._get_spec_segment(segment)
        print "Linear Combinations: {}".format(spec.__repr__())

        param_table_top = param_table.sort_values(by='rchisq').iloc[:ntop]
        params_out = smsyn.specmatch.lincomb(
            spec, pipe.libfile, pipe.wav_exclude, param_table_top
        )

        d = dict(params_out.ix[0])
        outstr = (
            "{teff:.0f} {logg:.2f} {fe:+.2f} {vsini:.2f} {psf:.2f} " +
            "{rchisq0:.2f} {rchisq1:.2f}"
        )
        outstr = outstr.format(**d)
        print outstr

        extname = 'lincomb_%i' % segment[0]
        smsyn.io.fits.write_dataframe(pipe.smfile, extname, params_out)

def polish(pipe):
    """
    Perform the polishing step using the lincomb step as a starting point
    """

    pipe.polish_output = {}
    for segment in pipe.segments:
        extname = 'lincomb_%i' % segment[0]
        params0 = smsyn.io.fits.read_dataframe(pipe.smfile, extname)
        assert len(params0)==1,"Must table of length 1"
        params0 = params0.iloc[0]
        lib = smsyn.library.read_hdf(pipe.libfile,wavlim=segment)
        spec = smsyn.io.spectrum.read_fits(pipe.smfile) # Save to output
        spec = spec[(segment[0]<spec['wav']) & (spec['wav']<segment[1])]

        print "Polishing fit to {}".format(spec.__repr__())

        wavmask = smsyn.specmatch.wav_exclude_to_wavmask(
            spec.wav, pipe.wav_exclude
        )    
        match = smsyn.match.Match(
            spec, lib, wavmask, cont_method='spline-fm',rot_method='rotmacro'
        )

        lm_params0 = lmfit.Parameters()
        lm_params0.add_many(
            ('teff',params0.teff),
            ('logg',params0.logg),
            ('fe', params0.fe),
            ('vsini', 10),
            ('psf', 1.5),
        )
        lm_params0['psf'].vary=False
        output = smsyn.specmatch.polish(
            match, lm_params0, angstrom_per_node=10, 
            objective_method='masked_nresid'
        )

        
        pipe.polish_output[segment[0]] = output


def read_pickle(pklfn,verbose=False):
    """
    Perform the polishing step using the lincomb step as a starting point
    """

    with open(pklfn,'r') as f:
        pipe = pickle.load(f)

    name, obs = pklfn.split('/')[-1][:-7].split('_')
    out = []
    for segment in pipe.segments:
        output = pipe.polish_output[segment[0]]
        keys = 'teff logg fe vsini psf'.split()
        s = ""
        d = {}
        d['name'] = name
        d['obs'] = obs
        d['segment0'] = segment[0]
        d['segment1'] = segment[1]
        for k in keys:
            s += "{:.2f} ".format(output['result'].params[k].value)
            d[k] = output['result'].params[k].value

        if verbose:
            print s
        out.append(d)

    out = pd.DataFrame(out)
    return out
