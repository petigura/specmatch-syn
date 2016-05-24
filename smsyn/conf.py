import os
from cStringIO import StringIO as sio
import pandas as pd
import cpsutils.kbc

SM_DIR = os.environ['SM_DIR']
kbc = cpsutils.kbc.loadkbc()
kbc['h5bname'] = kbc.name+"_"+kbc.obs+".h5"
kbc['rwpath'] = os.path.join(SM_DIR,"spectra/restwav/")
kbc['rwpath'] = kbc['rwpath']+ kbc['h5bname']


wloL_lc = [5200,5360,5530,6100,6210] # Linear combinations of best matches
wloL_fm = [5200,5360,5530,6100,6210] # Multi-segment forward modelling

def get_segdf():
    """
    Get Segment DataFrame
    
    Returns
    -------

    segdf : DataFrame with the wavelength regions [wlo,whi] and orders
            for the wavelength segments under consideration for SpecMatch.
    """

    s = """\
wlo   whi   ord
4980  5060  0
5055  5130  1
5125  5205  2
5160  5190  2
5200  5280  3
5280  5360  4
5360  5440  5
5445  5520  6
5530  5610  7
5620  5700  8
5710  5790  9
5800  5880  10
5900  5980  11
6000  6080  12
6100  6190  13
6150  6170  13
6159  6165  13
6210  6260  14
6320  6420  15"""

    segdf = pd.read_table(sio(s),sep=' ',comment='#', skipinitialspace=True)
    segdf = segdf.dropna()
    for k in 'wlo whi ord'.split():
        segdf[k] = segdf[k].astype(int)
        segdf.index = segdf.wlo
    return segdf

segdf = get_segdf()


# h5 directory with all the library spectra
