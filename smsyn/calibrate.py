"""Module contains class and functions to facilitate calibrating
specmatch results with touchstone stars.
"""
import numpy as np
import pandas as pd
import lmfit
from scipy.interpolate import LinearNDInterpolator

import smsyn.io.fits as smfits

class Calibrator(object):
    """
    Calibration object

    Args: 
        param (str): spectroscopic parameter for calibration
    """
    def __init__(self, param):
        self.param = param
        extname = {}
        extname['node_points'] = self.param+'_node_points'
        extname['node_values'] = self.param+'_node_values'
        self.extname = extname

    def fit(self, catalog, node_points, suffixes=['_sm','_lib']):
        """
        Peform a least squares minimization of linear interpolation paramete

        Args:
            catalog (pandas.DataFrame): Table with the (uncalibrated) specmatch
                parameters and the true stellar parameters determine through 
                some other method.
            node_points (pandas.DataFrame): Table of points at which to fit for
                calibration parameters.
            suffixes (list): Suffixes for the uncalibrated, and calibrate 
                parameters
        """
        self.param_uncal = self.param+suffixes[0]
        self.param_cal = self.param+suffixes[1]

        self.catalog = catalog
        self.suffixes = suffixes
        self.node_points = pd.DataFrame(node_points)

        node_values = np.zeros(len(node_points))
        self.node_values = pd.DataFrame({self.param:node_values})

        fit_params = lmfit.Parameters()
        for i in range(len(self.node_points)):
            fit_params.add('p%i' % i, value=self.node_values.ix[i,self.param])

        def resid(fit_params):
            for i in range(len(self.node_points)):
                self.node_values.ix[i,self.param] = fit_params['p%i' % i].value
            
            namemap = {}
            for k in 'teff logg fe'.split():
                namemap[k+suffixes[0]] = k

            params_uncal = self.catalog.rename(columns=namemap)
            _resid = (
                np.array(self.catalog[self.param_cal]).flatten() - 
                self.transform(params_uncal).flatten()
            )

            return _resid
        out = lmfit.minimize(resid, fit_params, method='fmin')
        lmfit.report_fit(out.params)

    def to_fits(self, fitsfn):
        """Save to fits file

        Args:
            fitsfn (str): Path to fits file

        Returns:
            None
        """
        extname = self.extname
        smfits.write_dataframe(fitsfn, extname['node_points'], self.node_points)
        smfits.write_dataframe(fitsfn, extname['node_values'], self.node_values)

    def transform(self, params_uncal):
        """Transform uncalibrated to calibrated parameters

        Args:
            params_uncal (dict): uncalibrated parameters with keys: 
                teff, logg, fe

        Return:
            dict: calibrated parameters
        """
        self.node_params = self.node_points.columns
        points = np.array(self.node_points)
        values = np.array(self.node_values)
        points_i = params_uncal[[k for k in self.node_params]]
        points_i = np.array(points_i)

        if points.shape[1]==1:
            points = points.flatten()
            values = values.flatten()
            idx = np.argsort(points)
            points = points[idx]
            values = values[idx]
            values_i = np.interp(points_i, points, values)
        else:
            interp = LinearNDInterpolator(points, values)
            values_i = interp(points_i)
            

        params_uncal['delta'] = values_i.reshape(-1)
        if params_uncal['delta'].isnull().sum() > 0:
            print params_uncal[params_uncal['delta'].isnull()]
        params_cal = np.array(params_uncal[self.param]) + values_i.reshape(-1)
        return params_cal

    def print_hyperplane(self, x, dx, fmt='.2f'):
        """Print hyperplane parameters.

        If the calibration is a function of N variables and there are
        N + 1 node points defined, then the calibration is a plane
        which can be defined in the following form:

            delta y = c0 + c1 * (x1 - x1_0)/(dx1) + ...

        Args:
             x (dict): value at which plane is defined.
             dx (slope): delta parameter used to define slope


        Returns:
             str: formatted string
             

        Example:
             >>> x = dict(teff=5500, logg=3.5, fe=0.0)
             >>> dx = dict(teff=100, logg=0.1, fe=0.1)
             >>> print_hyperplane(x, dx)
             $\Delta logg$ =

        """
        
        # Dimensions
        npoints, ndim = self.node_points.shape
        assert npoints==ndim+1, "Must have ndim + 1 points"

        # Convert to a length-1 data frame
        x = pd.DataFrame([x])
        dx = pd.DataFrame([dx])

        # Value of plane at reference point
        param_cal = self.transform(x)[0]
        param_uncal = x[self.param].iloc[0] 
        c0 = param_cal - param_uncal

        print ""*80
        print " {} Hyperplane Parameters ".format(self.param)
        s = "{:%s}" % fmt
        s = s.format(c0)
        i = 0 
        print "c%i" % i, s
        for k in 'teff logg fe'.split():
            i+=1
            if list(x.columns).count(k)==1:
                _x = x.copy()
                _x[k] += dx[k]
                param_cal = self.transform(_x)[0]
                param_uncal = _x[self.param].iloc[0] 
                
                slope = param_cal - param_uncal - c0
                s = "{:%s} * ({:s} - {:%s})/({:%s})" % (fmt, fmt, fmt) 
                s = s.format(
                    slope, k, x[k].iloc[0], 
                    dx[k].iloc[0] 
                )
                print "c%i" % i, s

def read_fits(fitsfn, param):
    """Restore calibrator object from fits file

    Args:
        fitsfn (str): path to fits file
        param (str): teff, logg, fe

    Returns:
        Calibtrator: object for performing calibration
    """
    cal = Calibrator(param)
    cal.node_points = smfits.read_dataframe(fitsfn, cal.extname['node_points'])
    cal.node_values = smfits.read_dataframe(fitsfn, cal.extname['node_values'])
    return cal



def calibrate(df, calfn, mode='uncal'):
    """
    Perform the calibration

    Args:
        df : DataFrame with teff, logg, fe
        calfn : Fits file that calibration parameters are saved to
        mode: `uncal` - put the uncalibrated parameters into _uncal suffix
        mode: `fivepane` - put the calibrated parameters to _sm
    """
    
    def namemap(suffix0,suffix1):
        _namemap = {}
        for k in 'teff logg fe'.split():
            _namemap[k+suffix0] = k+suffix1
        return _namemap

    if mode=='fivepane':
        # Replace _sm suffix with nothing
        df = df.rename(columns=namemap('_sm',''))
        
        # perform the calibrations
        for k in 'teff logg fe'.split():
            cal = read_fits(calfn,k)
            df[k+'_sm'] = cal.transform(df)

    elif mode=='uncal':
        for k in 'teff logg fe'.split():
            cal = read_fits(calfn,k)
            df[k+'_cal'] = cal.transform(df)

        # Move the teff -> teff_uncal
        df = df.rename(columns=namemap('','_uncal'))

        # Move the teff_cal -> teff
        df = df.rename(columns=namemap('_cal',''))

    else:
        assert False

    return df
