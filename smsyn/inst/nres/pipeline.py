"""Module that handels the fitting pipeline for hires
"""
import os
import numpy as np
import pandas as pd

import smsyn
import smsyn.io
import smsyn.io.spectrum
import smsyn.library
import smsyn.specmatch
from smsyn.inst.hires import shift
import smsyn.inst.hires.pipeline as hires_pipeline
from smsyn.inst.hires.pipeline import *

PACKAGE_DIR = os.path.dirname(smsyn.__file__)

HiresPipeline = hires_pipeline.Pipeline

class Pipeline(HiresPipeline):
    """Pipeline object

    Top level controller for the SpecMatch pipeline.

    Args:
        smfile (str): path to input spectrum
        libfile (str): path to library file
        segfile (Optional[str]): path to csv file that stores the segment 
            begining and endings. If None, it's read in from smsyn.inst.hires
        wav_excludefile (Optionla[str]): path to csv file that stores
            the wavelength regions that we exclude in our fits. If
            None, it's read in from symsyn.libraries
    """

    def __init__(self, smfile, libfile, segfile=None, wav_excludefile=None):
        if segfile is None:
            segfile = os.path.join(PACKAGE_DIR,'inst/apf/segments.csv')
        if wav_excludefile is None:
            wav_excludefile = os.path.join(
                PACKAGE_DIR,'models/coelho05/coelho05_wavmask.csv'
            )

        wav_exclude = pd.read_csv(wav_excludefile, comment='#')
        wav_exclude = np.array(wav_exclude)
        segments = pd.read_csv(segfile, comment='#')
        segments = np.array(segments)

        self.smfile = smfile
        self.pklfn = smfile.replace('.fits', '.pkl')
        self.libfile = libfile
        self.wav_exclude = wav_exclude
        self.segments = segments

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
        wavmask = wavmask | mask_spectrum(spec.wav, spec.flux)

        match = smsyn.match.Match(
            spec, lib, wavmask, cont_method='spline-fm',rot_method='rotmacro'
        )

        # Attach parameters from previous step. Starting at a low value of
        # vsini seems to help convergence
        lm_params0 = lmfit.Parameters()
        lm_params0.add_many(
            ('teff', params0.teff),
            ('logg', params0.logg),
            ('fe',  params0.fe),
            ('vsini', 0.1),
        )
        output = smsyn.specmatch.polish(
            match, lm_params0, 1.1, 0.4, angstrom_per_node=10,
        )

        pipe.polish_output[segment[0]] = output


def read_pickle(pklfn,verbose=False):
    """
    Read the parameters from the pickle save function
    """

    with open(pklfn,'r') as f:
        pipe = pickle.load(f)

    fsplit = pklfn.split('/')[-1][:-7].split('_')
    if len(fsplit) == 2:
        name, obs = fsplit
    else:
        name = '_'.join(fsplit[:-1])
        obs = fsplit[-1]

    out = []
    for segment in pipe.segments:
        output = pipe.polish_output[segment[0]]
        s = ""
        d = {}
        d['name'] = name
        d['obs'] = obs
        d['segment0'] = segment[0]
        d['segment1'] = segment[1]
        d['method'] = 'polish'
        d = dict(d, **output['params_out'])
        out.append(d)

    for segment in pipe.segments:
        extname = 'lincomb_%i' % segment[0]
        d = smsyn.io.fits.read_dataframe(pklfn.replace('pkl','fits'),extname)
        d = dict(d.iloc[0])
        d['name'] = name
        d['obs'] = obs
        d['segment0'] = segment[0]
        d['segment1'] = segment[1]
        d['method'] = 'lincomb'
        out += [d]

    out = pd.DataFrame(out)

    return pipe, out
