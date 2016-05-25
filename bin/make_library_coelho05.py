#!/usr/bin/env python
import glob
import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from astropy.io import fits

import smsyn.library

def main(fitsdir, outfile):
    fitsfiles = glob.glob(os.path.join(fitsdir,'*.fits'))
    model_table = []
    model_spectra = []

    for i, fitsfn in enumerate(fitsfiles):
        # memmap=False prevents "too many open files" error
        with fits.open(fitsfn, memmap=False) as hduL:
            header = hduL[0].header

            table_row = dict(
                teff = header['TEFF'],
                logg = header['LOG_G'],
                fe = header['FEH'],
                afe = header['AFE']
            )
            model_table.append(table_row)
            s = hduL[0].data[0]
            model_spectra.append(s)
            del hduL

    w0 = header['CRVAL1'] # Starting wavelength (Angstroms)
    dw = header['CD1_1'] # Wavelenth sampling (Angstroms)
    wavelength = w0 + np.arange(s.size) * dw

    model_spectra = np.vstack(model_spectra)
    model_table = pd.DataFrame(model_table)
    header = dict(
        model_name='coelho05',
        model_reference='Coelho et al. 2005'
    )
    lib = smsyn.library.Library(
        header, model_table, wavelength, model_spectra
    )
    lib.to_hdf(outfile)
    print "created {}".format(outfile)

if __name__=="__main__":
    psr = ArgumentParser(description="Create the coelho model library")
    psr.add_argument('fitsdir', type=str, help="Path to coelho .fits files")
    psr.add_argument('outfile', type=str, help="Path to hdf output file")
    args  = psr.parse_args()
    main(args.fitsdir, args.outfile)

