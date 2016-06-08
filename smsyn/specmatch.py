"""Top-level fitting and polishing routines

"""
import numpy as np
import pandas as pd
import lmfit
import smsyn.io.spectrum
import smsyn.library
import smsyn.match
import smsyn.io.fits

def grid_search(spec0, libfile, segment, wav_exclude, param_table, idx_coarse, 
                idx_fine):
    """
    Args:
        segment (list): upper and lower bounds of segment
        libfile (str): path to library hdf5 file. 
        wav_exclude (list): define wavlengths to exclude from fit
            e.g. [[5018, 5019.5],[5027.5, 5028.5]] 
    """
    spec = spec0.copy()
    lib = smsyn.library.read_hdf(libfile,wavlim=segment)
    spec = spec[(segment[0] < spec.wav) & (spec.wav < segment[1])]
    wavmask = np.zeros_like(spec.wav).astype(bool) # Default: no points masked
    nwav_exclude = len(wav_exclude)
    for i in range(nwav_exclude):
        wav_min, wav_max = wav_exclude[i]
        wavmask[(wav_min < spec.wav) & (spec.wav < wav_max)] = True

    match = smsyn.match.Match(spec, lib, wavmask)
    
    # First do a coarse grid search
    node_wav = spline_nodes(match.spec.wav[0],match.spec.wav[-1])
    for _node_wav in node_wav:
        param_table['sp%d' % _node_wav] = 1.0

    param_table_coarse = grid_search_loop(match, param_table.ix[idx_coarse])

    # For the fine grid search, 
    top = param_table_coarse.sort_values(by='rchisq').head(10) 
    tab = param_table.ix[idx_fine]
    tab = tab.drop(idx_coarse)

    param_table_fine = tab[
        tab.teff.between(top.teff.min(),top.teff.max()) & 
        tab.logg.between(top.logg.min(),top.logg.max()) & 
        tab.fe.between(top.fe.min(),top.fe.max()) 
    ]
    param_table_fine = grid_search_loop(match, param_table_fine)
    param_table = pd.concat([param_table_coarse, param_table_fine])
    return param_table

def grid_search_loop(match, param_table0):
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
    nrows = len(param_table0)
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

    print_grid_search()
    counter=0
    for i, row in param_table.iterrows():
        for key in param_keys:
            params[key].set(row[key])

        mini = lmfit.minimize(match.nresid, params, maxfev=100)
        
        for key in mini.var_names:
            param_table.loc[i, key] = mini.params[key].value
        param_table.loc[i,'chisq'] = mini.chisqr
        param_table.loc[i,'rchisq'] = mini.redchi
        param_table.loc[i,'nfev'] = mini.nfev

        nresid = match.masked_nresid( mini.params )
        logprob = -0.5 * np.sum(nresid**2) 
        param_table.loc[i,'logprob'] = logprob
        d = dict(param_table.loc[i])
        d['counter'] = counter 
        d['nrows'] = nrows
        print_grid_search(d)
        counter+=1
    return param_table

def print_grid_search(*args):
    if len(args)==0:
        print "          {:4s} {:4s} {:3s} {:4s} {:6s} {:4s}".format(
            'teff','logg','fe','vsini','rchisq','nfev'
        )
    if len(args)==1:
        d = args[0]
        print "{counter:4d}/{nrows:4d} {teff:4.0f} {logg:4.1f} {fe:+2.1f} {vsini:3.1f}  {rchisq:6.2f} {nfev:4.1f}".format(**d)


def spline_nodes(wav_min, wav_max, angstrom_per_node=20,):
    # calculate number of spline nodes
    node_wav_min = np.floor(wav_min)
    node_wav_max = np.ceil(wav_max)
    nodes = (node_wav_max - node_wav_min) / angstrom_per_node
    nodes = int(np.round(nodes))
    node_wav = np.linspace(node_wav_min, node_wav_max, nodes)
    node_wav = node_wav.astype(int)
    return node_wav

def add_spline_nodes(params, node_wav, vary=True):
    for node in nodes:
        params.add('sp%i' % node,value=1.0, vary=vary)

def add_model_weights(params, nmodels):
    value = 1.0 / nmodels
    for i in range(nmodels):
        param.add('p%i' % i ,value=value,min=0,max=1)

def get_model_weights(params):
    nmodels = len([k for k in params.keys() if k[:2]=='mw'])
    model_weights = [params['mw%i' % i].value for i in range(nmodels)]
    model_weights = np.array(model_weights)
    return model_weights


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
