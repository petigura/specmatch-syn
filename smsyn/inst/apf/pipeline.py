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

PACKAGE_DIR = os.path.dirname(smsyn.__file__)

HiresPipeline = hires_pipeline.Pipeline

class APFPipeline(HiresPipeline):
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
