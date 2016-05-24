"""
Module to augment pandas functionality. If time, make a pull request.
"""
import pandas as pd
import numpy as np
from cStringIO import StringIO
import os

def LittleEndian(r):
    names = r.dtype.names
    data = {}
    for n in names:
        if r[n].dtype.byteorder=='>':
            data[n] = r[n].byteswap().newbyteorder() 
        else:
            data[n] = r[n] 

    r = np.rec.fromarrays(data.values(),names=data.keys())
    return r

def df_to_ndarray(df):
    """
    Convert Pandas DataFrame to ndarray
    
    If there are objects in the array, convert them to a string type.

    Parameters
    ----------
    df  : DataFrame
    res : numpy ndarray
    """

    arrayList = []
    for c in df.columns:
        if df[c].dtype==np.dtype('O'):
            arr = np.array(df[c]).astype(str)
        else:
            arr = np.array(df[c])
        arrayList += [arr]

    res = np.rec.fromarrays(arrayList,names=list(df.columns))
    res = np.array(res)
    return res  


def string_to_df(s):
    """
    String to DataFrame

    A little convienence function to make a data frame from space-
    separated string.

    Tables fields are separated by a space, and can be commented out with a #

    wlo   whi   ord
    4980  5060  0
    5055  5130  1
    #5125  5205  2
    """
    df = pd.read_table(StringIO(s),comment='#',sep='\s*',engine='python')
    df = df.dropna()
    return df
