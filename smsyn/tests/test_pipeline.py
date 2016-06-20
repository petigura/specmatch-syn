#!/usr/bin/env python 
import os
from argparse import ArgumentParser
from smsyn.inst.hires.pipeline import Pipeline
from smsyn import DATA_DIR

if __name__=="__main__":
    psr = ArgumentParser() 
    psr.add_argument('libfile',type=str,
                     help="Path to model library in h5 format")
    psr.add_argument('specfile',type=str,
                     help="Path to 1D input spectrum file on the rest wavelength scale. \
                     The results of the pipeline will also be stored in this fits file.")
    args = psr.parse_args()
    outfile = args.specfile
    
    pipe = Pipeline(outfile, args.libfile)
    pipe.grid_search(debug=False)
    pipe.lincomb()
