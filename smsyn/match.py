
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline, UnivariateSpline

class Match(object):

    def __init__(self, spec, lib, wavmask):
        """

        
        
        Args:
            spec (smsyn.spectrum.Spectrum): Spectrum object containing
                the data to be fit
            lib (smsyn.library.Library): Library object containing the
                model library and `synth` method

            wavmask (boolean array): same length as spec.wav. If false ignore
                in the likelihood calculation
        """

        self.spec = spec
        self.lib = lib
        self.wavmask = wavmask


    def model(self, params, wav=None, **kwargs):
        """Calculate model

        Return the model for a given set of parameters

        Args:
            params (lmfit.Parameters): Parameters object containing
               at least teff, logg, fe, vsini, psf, and spline coefficients
            wav (array): (optional) array of wavelengths at which to calculate
                the model. Useful for generating a more finely sampled
                model for plotting
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
        mcont = self.continuum(wav, _model, _model)
        _model /= mcont
        
        return _model

    
    def continuum(self, wav, data, weights, s=40):
        """Continuum model

        Return only the model for the continuum for a given set of
        parameters.
        
        Args:
            params (lmfit.Parameters): See params in self.model
            wav: array of wavelengths at which to calculate
               the continuum model.

        Returns:
            array: continuum model
               
        """

        splrep = UnivariateSpline(wav, data, w=weights, s=40)
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

        flux = self.spec.flux / self.continuum(self.spec.wav, self.spec.flux, self.spec.flux)
        
        res = flux - self.model(params)

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

    
