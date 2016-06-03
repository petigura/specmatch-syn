"""Module to define the Spectrum class

This module defines the Spectrum object used for specmatch-synth

Attributes:
    FITSCOLDEFS (list of tuples): column labels and metadata for output fits files
    
    
"""

from astropy.io import fits

FITSCOLDEFS = [
           ('wav', 'D', 'Wavelength', 'Angstroms'),
           ('flux', 'D', 'Normalized flux', 'relative intensity'),
           ('uflux', 'D', 'Flux uncertainty', 'relative intensity')
]


class Spectrum(object):
    """Spectrum object

    This class defines the Spectrum object used to
    handle data and associated metadata.

    Args:
        wav (array): wavelengths corresponding to each pixel in the
            flux array
        flux (array): continuum-normalized flux as a function of
            rest wavelength
        uflux (array): relative flux uncertainty
        header (dict): dictionary containing metadata associated with the
            observed spectrum. Similar to a header from a fits file.
            Required keys: OBJECT, RESOLUTION, DATE-OBS

    """
        
    def __init__(self, wav, flux, uflux, header):
        self.wav = wav
        self.flux = flux
        self.uflux = uflux
        self.header = header

    def to_fits(self, outfile, clobber=True):
        """Save to FITS

        Save a Spectrum object as a mutli-extension fits file.

        Args:
            outfile (string): name of output file name
            clobber (bool): if true, will overwrite existing file
            
        """
        
        columns = []
        for i,col in enumerate(FITSCOLDEFS):
            colinfo = FITSCOLDEFS[i]
            coldata = self.__dict__[colinfo[0]]
            fitscol = fits.Column(array=coldata, format=colinfo[1], name=colinfo[0], unit=colinfo[3])

            columns.append(fitscol)

        table_hdu = fits.BinTableHDU.from_columns(columns)

        fitsheader = fits.Header()
        fitsheader.update(self.header)

        primary_hdu = fits.PrimaryHDU(header=fitsheader)
        
        hdu_list = fits.HDUList([primary_hdu, table_hdu])
        
        hdu_list.writeto(outfile, clobber=clobber)


def read_fits(filename):
    """Read spectrum from fits file

    Read in a spectrum as saved by the Spectrum.to_fits method into
    a Spectrum object

    Args:
        filename (string): path to h5 file that contains the grid
            of stellar atmosphere models
        
    Returns:
        Spectrum object
        
    """
    
    hdu = fits.open(filename)
    header = hdu[0].header
    table = hdu[1].data

    record_names = table.dtype.names
    required_cols = [k[0] for k in FITSCOLDEFS]
    for k in required_cols:
        assert k in record_names, "Column {0} not found. {1} are all \
        requried columns in the fits table.".format(k, required_cols)

    spec = Spectrum(table['wav'], table['flux'], table['uflux'], header)

    return spec

