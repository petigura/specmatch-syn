import pandas as pd
import numpy as np
from cStringIO import StringIO
import os
import astropy.io.fits 
import smsyn.io.fits
from astropy.table import Table
import os
def write_hdu(fitsfn, extname, hdu):
    """

    """
    if os.path.exists(fitsfn) is False:
        primary_hdu = astropy.io.fits.PrimaryHDU()
        astropy.io.fits.append(
            fitsfn, primary_hdu.data, header=primary_hdu.header
        )

    data = hdu.data
    hdu.header['EXTNAME'] = extname
    header = hdu.header
    try:
        astropy.io.fits.update(fitsfn, data, extname, header=header)
    except KeyError:
        astropy.io.fits.append(fitsfn, data, header=header)

def little_endian(inp):
    names = inp.dtype.names
    data = {}
    for n in names:
        if inp[n].dtype.byteorder=='>':
            data[n] = inp[n].byteswap().newbyteorder() 
        else:
            data[n] = inp[n] 

    out = np.rec.fromarrays(data.values(),names=data.keys())
    return out

def write_dataframe(fitsfn, extname, data):
    # Weird object converion needed to prevent type errors. Don't
    # really understand what was going on here, perhaps some issue
    # with unicode.
    data = Table.from_pandas(data)
    data = data.as_array()
    hdu = astropy.io.fits.BinTableHDU(data=data)
    write_hdu(fitsfn, extname, hdu,)
    return None

def read_dataframe(fitsfn, extname):
    hdu = astropy.io.fits.open(fitsfn)[extname]
    data = pd.DataFrame(little_endian(hdu.data))    
    return data 
