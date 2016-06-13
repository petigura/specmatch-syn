"""Module that handels the fitting pipeline for hires
"""
import os

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

    Top level controller for the SpecMatch pipeline.

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
        self.libfile = libfile
        self.wav_exclude = wav_exclude
        self.segments = segments

    def _get_spec_segment(self, segment):
        spec = smsyn.io.spectrum.read_fits(self.smfile)
        spec = spec[(segment[0] < spec.wav) & (spec.wav < segment[1])]
        return spec

    def grid_search(self, debug=False):
        # Load up the model library
        lib = smsyn.library.read_hdf(self.libfile, wavlim=[4000,4100])
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
            spec = self._get_spec_segment(segment)
            param_table = smsyn.specmatch.grid_search(
                spec, self.libfile, segment, self.wav_exclude, param_table, 
                idx_coarse, idx_fine
            )

            extname = 'grid_search_%i' % segment[0]
            smsyn.io.fits.write_dataframe(self.smfile, extname,param_table,)
            
    def lincomb(self):
        ntop = 8
        print "teff logg fe vsini psf rchisq0 rchisq1"

        for segment in self.segments:
            extname = 'grid_search_%i' % segment[0]
            param_table = smsyn.io.fits.read_dataframe(self.smfile, extname)
            param_table_top = param_table.sort_values(by='rchisq').iloc[:ntop]
            model_indecies = np.array(param_table_top.model_index.astype(int))
            spec = self._get_spec_segment(segment)
            lib = smsyn.library.read_hdf(self.libfile, wavlim=segment)

            wavmask = np.zeros_like(spec.wav).astype(bool)
            match = smsyn.match.MatchLincomb(spec, lib, wavmask, model_indecies)

            params = lmfit.Parameters()
            nodes = smsyn.specmatch.spline_nodes(
                spec.wav[0],spec.wav[-1], angstroms_per_node=10
            )
            smsyn.specmatch.add_spline_nodes(params, nodes)
            smsyn.specmatch.add_model_weights(params, ntop, min=0.01)
            params.add('vsini',value=5)
            params.add('psf',value=1, vary=False)

            out = lmfit.minimize(match.masked_nresid, params)
            def rchisq(params):
                nresid = match.masked_nresid(params)
                return np.sum(nresid**2) / len(nresid)

            rchisq0 = rchisq(params)
            rchisq1 = rchisq(out.params)

            #print lmfit.fit_report(out.params)

            mw = smsyn.specmatch.get_model_weights(out.params)
            mw = np.array(mw)
            mw /= mw.sum()

            params_out = lib.model_table.iloc[model_indecies]
            params_out = params_out['teff logg fe'.split()]
            params_out = pd.DataFrame((params_out.T * mw).T.sum()).T
            
            d = dict(params_out.ix[0])
            d['vsini'] = out.params['vsini'].value
            d['psf'] = out.params['psf'].value
            d['rchisq0'] = rchisq0
            d['rchisq1'] = rchisq1
            outstr = (
                "{teff:.0f} {logg:.2f} {fe:+.2f} {vsini:.2f} {psf:.2f} " +
                "{rchisq0:.2f} {rchisq1:.2f}"
            )
            outstr = outstr.format(**d)
            print outstr

            extname = 'lincomb_%i' % segment[0]
            smsyn.io.fits.write_dataframe(self.smfile, extname, params_out)

