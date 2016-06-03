"""

This module defines the Match class that is used in fitting routines.

"""

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

class Match(object):
    def __init__(self, spec, lib, wavmask):
        """

        The Match object used for fitting functions
        
        Args:
            spec (smsyn.spectrum.Spectrum): Spectrum object containing
                the data to be fit
            lib (smsyn.library.Library): Library object containing
                the model library and `synth` method

            wavmask (boolean array): same length as spec.wav. If false
                ignore in the likelihood calculation
                
        """

        self.spec = spec
        self.lib = lib
        self.wavmask = wavmask

    def model(self, params, wav=None, **kwargs):
        """Calculate model

        Return the model for a given set of parameters

        Args:
            params (lmfit.Parameters): Parameters object containing at least
                teff, logg, fe, vsini, psf, and spline coefficients
            wav (array): (optional) array of wavelengths at which to
                calculate the model. Useful for generating a more finely
                sampled model for plotting
            **kwargs: extra keyword arguments passed to lib.synth
            
        """

        if wav is None:
            wav = self.spec.wav

        teff = params['teff'].value
        logg = params['logg'].value
        fe = params['fe'].value
        vsini = params['vsini'].value
        psf = params['psf'].value

        _model = self.lib.synth(wav, teff, logg, fe, vsini, psf, **kwargs)
        _model *= self.continuum(params, wav)
        return _model

    def continuum(self, params, wav):
        """Continuum model

        Return only the model for the continuum for a given set of parameters.
        
        Args:
            params (lmfit.Parameters): See params in self.model
            wav: array of wavelengths at which to calculate the continuum model.

        Returns:
            array: continuum model
               
        """

        node_wav = []
        node_flux = []
        for key in params.keys():
            if key.startswith('sp'):
                node_wav.append( float( key.replace('sp','') ) )
                node_flux.append( params[key].value )

        assert len(node_wav) > 3 and len(node_flux) > 3, \
            "Too few spline nodes for the continuum model."
            
        node_wav = np.array(node_wav)
        node_flux = np.array(node_flux)
        splrep = InterpolatedUnivariateSpline(node_wav, node_flux)
        cont = splrep(wav)
        return cont
        
    def resid(self, params):
        """Residuals

        Return the residuals

        Args:
            params (lmfit.Parameters): see params in self.model

        Returns:
            array: model minus data

        """
        
        res = self.spec.flux - self.model(params, wav=self.spec.wav) 
        return res

    def nresid(self, params):
        """Normalized residuals

        Args:
            params (lmfit.Parameters): see params in self.model

        Returns:
            array: model minus data divided by errors

        """

        return self.resid(params) / self.spec.uflux

    def masked_nresid(self, params):
        """Masked normalized residuals

        Return the normalized residuals multiplied by the
        boolean masked defined in self.spec.wavmask

        Args:
            params  (lmfit.Parameters): see params in self.model

        Returns:
            array: normalized residuals where self.wavmask == 1

        """

        return self.nresid(params)[self.wavmask]

    def chi2med(self, params):
        _resid = self.resid(params)
        med = np.median(_resid)
        _resid -= med
        _chi2med = np.sum(_resid**2)
        return _chi2med