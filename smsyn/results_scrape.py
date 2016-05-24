import h5py
import pandas as pd

from smsyn import h5plus

basedkeys = """
name obs sig DECKNAME snr 
restwav_stop_time restwav_sha
telluric_stop_time telluric_sha
specmatch_stop_time specmatch_sha""".split()

polishdkeys = "teff logg fe vsini chi polish_stop_time polish_sha".split()

def get_attrs(h5,keys):
    """
    Convenience function to return dictionary from attribute list,
    handling missing items gracefully
    """
    attrs = h5.attrs
    out_dict = dict(
        [(k,attrs[k]) for k in keys if attrs.keys().count(k)>0 ]
    )

    return out_dict

def polish(h5file):
    """
    For a results h5 file that has been processed with polish, pull
    all the attributes, from the root group.
    """
    with h5py.File(h5file) as h5:
        based = get_attrs(h5,basedkeys)
        based['exp'] = h5.attrs['FRAMENO']
        based['run'] = h5.attrs['run']
        based['runnum'] = int(h5.attrs['run'][1:])

        g = h5['fm']
        outd = dict(based,**get_attrs(g,polishdkeys))
    return outd
    
