"""Top-level fitting and polishing routines

"""
import numpy as np
import lmfit
import pandas as pd
import smsyn.spectrum
import smsyn.library

def grid_search(match, param_table0):
    """Grid Search

    Perform grid search using starting values listed in a parameter table.

    Args:
        match (smsyn.match.Match): `Match` object.
        param_table0 (pandas DataFrame): Table defining the parameters to search
            over.

    Returns:
        pandas DataFrame: results of the grid search with the input parameters
            and the following columns added: `logprob` log likelihood, `chisq`
            chi-squared, and `rchisq` reduced chisq, `niter` number of 
            iterations
    """

    param_keys = param_table0.columns    
    param_table = param_table0.copy()
    for col in 'chisq rchisq logprob nfev'.split():
        param_table[col] = np.nan

    params = lmfit.Parameters()
    for key in param_keys:
        params.add(key)
        params[key].vary = False
        if key[:2] == 'sp':
            params[key].vary = True

    params['vsini'].vary = True

    for i, row in param_table.iterrows():
        for key in param_keys:
            params[key].set(row[key])

        mini = lmfit.minimize(match.nresid, params)
        
        for key in mini.var_names:
            param_table.loc[i, key] = mini.params[key].value
        param_table.loc[i,'chisq'] = mini.chisqr
        param_table.loc[i,'rchisq'] = mini.redchi
        param_table.loc[i,'nfev'] = mini.nfev

        nresid = match.masked_nresid( mini.params )
        logprob = -0.5 * np.sum(nresid**2) 
        param_table.loc[i,'logprob'] = logprob


        print pd.DataFrame(param_table.loc[i]).T

    return param_table

def make_matchlist(spec0, libpath, wavmask0, wavlims):
    matchlist = []
    for wavlim in wavlims:
        lib = smsyn.library.read_hdf(libpath,wavlim=wavlim)
        b = (wavlim[0] < spec0.wav) & (spec0.wav < wavlim[1])                

        spec = smsyn.spectrum.Spectrum(
            spec0.wav[b], spec0.flux[b], spec0.uflux[b], spec0.header
        )
        wavmask = wavmask0[b]
        match = smsyn.match.Match(spec, lib, wavmask)
        matchlist.append(match)
    
    return matchlist


def grid_search(match, param_table0):
    """Grid Search

    Perform grid search using starting values listed in a parameter table.

    Args:
        match (smsyn.match.Match): `Match` object.
        param_table0 (pandas DataFrame): Table defining the parameters to search
            over.

    Returns:
        pandas DataFrame: results of the grid search with the input parameters
            and the following columns added: `logprob` log likelihood, `chisq`
            chi-squared, and `rchisq` reduced chisq, `niter` number of 
            iterations
    """

    param_keys = param_table0.columns    
    param_table = param_table0.copy()
    for col in 'chisq rchisq logprob nfev'.split():
        param_table[col] = np.nan

    params = lmfit.Parameters()
    for key in param_keys:
        params.add(key)
        params[key].vary = False
        if key[:2] == 'sp':
            params[key].vary = True


    for i, row in param_table.iterrows():
        for key in param_keys:
            params[key].set(row[key])

        params['vsini'].vary = True
        params['vsini'].min = 0.5

        mini = lmfit.minimize(match.nresid, params)
        
        for key in mini.var_names:
            param_table.loc[i, key] = mini.params[key].value
        param_table.loc[i,'chisq'] = mini.chisqr
        param_table.loc[i,'rchisq'] = mini.redchi
        param_table.loc[i,'nfev'] = mini.nfev

        nresid = match.masked_nresid( mini.params )
        logprob = -0.5 * np.sum(nresid**2) 
        param_table.loc[i,'logprob'] = logprob


        print pd.DataFrame(param_table.loc[i]).T

    return param_table

def make_matchlist(spec0, libpath, wavmask0, wavlims):
    matchlist = []
    for wavlim in wavlims:
        lib = smsyn.library.read_hdf(libpath,wavlim=wavlim)
        b = (wavlim[0] < spec0.wav) & (spec0.wav < wavlim[1])                

        spec = smsyn.spectrum.Spectrum(
            spec0.wav[b], spec0.flux[b], spec0.uflux[b], spec0.header
        )
        wavmask = wavmask0[b]
        match = smsyn.match.Match(spec, lib, wavmask)
        matchlist.append(match)
    
    return matchlist

def polish(matchlist, params0, angstrom_per_node=20, objective_method='chi2med'):
    """Polish parameters
    
    Given a list of match object, polish the parameters segment by segment

    Args:
        angstrom_per_node (float): approximate separation between continuum and
            spline nodes. Number of nodes will be rounded to nearest integer.

    """

    nmatch = len(matchlist)

    output = []
    
    for i in range(nmatch):
        match = matchlist[i]
        params = lmfit.Parameters()
        for name in params0.keys():
            params.add(name)
            params[name].value = params0[name].value
            params[name].vary = params0[name].vary
            params[name].min = params0[name].min
            params[name].max = params0[name].max

        params['vsini'].min = 0.5

        # calculate number of spline nodes
        node_wav_min = np.floor(match.spec.wav[0])
        node_wav_max = np.ceil(match.spec.wav[-1])
        nodes = (node_wav_max - node_wav_min) / angstrom_per_node
        nodes = int(np.round(nodes))
        node_wav = np.linspace(node_wav_min, node_wav_max, nodes)
        for _node_wav in node_wav:
            key = 'sp%d' % _node_wav
            params.add(key)
            params[key].value = 1.0

        objective = getattr(match,objective_method)
        def iter_cb(params, iter, resid):
            pass
        out = lmfit.minimize(
            objective, params, method='nelder', iter_cb=iter_cb
        )

        resid = match.resid(out.params)
        medresid = np.median(resid)
        resid -= medresid
        d = dict(
            result=out, model=match.model(out.params), 
            continuum=match.continuum(out.params, match.spec.wav), 
            wav=match.spec.wav, resid=resid, 
            objective=objective(out.params)
        )
        output.append(d)

    return output
