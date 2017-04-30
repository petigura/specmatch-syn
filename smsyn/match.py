"""

This module defines the Match class that is used in fitting routines.

"""

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import LSQUnivariateSpline
from scipy.interpolate import splrep, spleval

class Match(object):
    def __init__(self, *args, **kwargs):
        """

        The Match object used for fitting functions
        
        Args:
            spec (smsyn.spectrum.Spectrum): Spectrum object containing
                the data to be fit
            lib (smsyn.library.Library): Library object containing
                the model library and `synth` method

            wavmask (boolean array): same length as spec.wav. If True
                ignore in the likelihood calculation
            cont_method (str): Method for accounting for the mismatch in the 
                continuum between observed and model spectra. One of:
                `spline-forward-model` or `spline-double-div`
        """
        spec, lib, wavmask = args
        
        assert wavmask.dtype==np.dtype('bool'), "mask must be boolean"

        self.spec = spec
        self.lib = lib
        self.wavmask = wavmask

        assert kwargs.has_key('cont_method'), "Must contain cont_method"
        self.cont_method = kwargs['cont_method']

        assert kwargs.has_key('rot_method'), "Must contain rot_method"
        self.rot_method = kwargs['rot_method']

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

        _model = self.lib.synth(
            wav, teff, logg, fe, vsini, psf, self.rot_method, **kwargs
        )
        return _model

    def spline(self, params, wav):
        """Continuum model

        Unpacks the params object and returns a spline evaluated at specified
        wavelengths.
        
        Args:
            params (lmfit.Parameters): See params in self.model
            wav: array of wavelengths at which to calculate the continuum model.

        Returns:
            array: spline
               
        """

        node_wav, node_flux = get_spline_nodes(params)
        assert len(node_wav) > 3 and len(node_flux) > 3, "Must pass > 3 nodes"
            
        node_wav = np.array(node_wav)
        node_flux = np.array(node_flux)
        splrep = InterpolatedUnivariateSpline(node_wav, node_flux)
        cont = splrep(wav)
        return cont
        
    def resid(self, params, **kwargs):
        """Residuals

        Return the residuals

        Args:
            params (lmfit.Parameters): see params in self.model

        Returns:
            array: model minus data

        """
        flux = self.spec.flux.copy()
        wav = self.spec.wav
        model = self.model(params, wav=wav, **kwargs) 

        if self.cont_method=='spline-fm':
            model /= self.spline(params, wav)

        if self.cont_method=='spline-dd':
            node_wav, node_flux = get_spline_nodes(params)
            t = node_wav[1:-1]

            spl = LSQUnivariateSpline(wav, flux, t, k=3, ext=0)
            flux /= spl(wav)

            spl = LSQUnivariateSpline(wav, model, t, k=3, ext=0)
            model /= spl(wav)


        res = flux - model
        return res

    def nresid(self, params, **kwargs):
        """Normalized residuals

        Args:
            params (lmfit.Parameters): see params in self.model

        Returns:
            array: model minus data divided by errors

        """

        return self.resid(params, **kwargs) / self.spec.uflux

    def masked_nresid(self, params, **kwargs):
        """Masked normalized residuals

        Return the normalized residuals with masked wavelengths excluded

        Args:
            params  (lmfit.Parameters): see params in self.model

        Returns:
            array: normalized residuals where self.wavmask == 1

        """

        _out = self.nresid(params, **kwargs)[~self.wavmask]
        return _out

    def chi2med(self, params):
        _resid = self.resid(params)
        med = np.median(_resid)
        _resid -= med
        _chi2med = np.sum(_resid**2)
        return _chi2med


class MatchLincomb(Match):
    def __init__(self, *args, **kwargs):
        """

        The Match object used for fitting functions
        
        Args:
            spec (smsyn.spectrum.Spectrum): Spectrum object containing
                the data to be fit
            lib (smsyn.library.Library): Library object containing
                the model library and `synth` method

            wavmask (boolean array): same length as spec.wav. If True
                ignore in the likelihood calculation
            cont_method (str): Method for accounting for the mismatch in the 
                continuum between observed and model spectra. One of:
                `spline-forward-model` or `spline-double-div`
        """
        spec, lib, wavmask, model_indecies = args
        
        assert wavmask.dtype==np.dtype('bool'), "mask must be boolean"

        self.spec = spec
        self.lib = lib
        self.wavmask = wavmask
        self.model_indecies= model_indecies

        cont_method = 'spline-fm'
        if kwargs.has_key('cont_method'):
            cont_method = kwargs['cont_method']
        self.cont_method = cont_method

    def model(self, params, wav=None):
        if wav is None:
            wav = self.spec.wav

        mw = get_model_weights(params)
        vsini = params['vsini'].value
        psf = params['psf'].value
        _model = self.lib.synth_lincomb(
            wav, self.model_indecies, mw, vsini, psf
        )
        return _model


def spline_nodes(wav_min, wav_max, angstroms_per_node=20,):
    # calculate number of spline nodes
    node_wav_min = np.floor(wav_min)
    node_wav_max = np.ceil(wav_max)
    nodes = (node_wav_max - node_wav_min) / angstroms_per_node
    nodes = int(np.round(nodes))
    node_wav = np.linspace(node_wav_min, node_wav_max, nodes)
    node_wav = node_wav.astype(int)
    return node_wav

def add_spline_nodes(params, node_wav, vary=True):
    for node in node_wav:
        params.add('sp%i' % node, value=1.0, vary=vary)

def get_spline_nodes(params):
    node_wav = []
    node_flux = []
    for key in params.keys():
        if key.startswith('sp'):
            node_wav.append( float( key.replace('sp','') ) )
            node_flux.append( params[key].value )
    node_wav = np.array(node_wav) 
    node_flux = np.array(node_flux) 
    return node_wav, node_flux

def add_model_weights(params, nmodels, min=0, max=1):
    value = 1.0 / nmodels
    for i in range(nmodels):
        params.add('mw%i' % i ,value=value,min=min,max=max)

def get_model_weights(params):
    nmodels = len([k for k in params.keys() if k[:2]=='mw'])
    model_weights = [params['mw%i' % i].value for i in range(nmodels)]
    model_weights = np.array(model_weights)
    return model_weights


