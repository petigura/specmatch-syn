"""Top-level fitting and polishing routines

"""
import numpy as np
import pandas as pd
import lmfit
import smsyn.io.spectrum
import smsyn.library
import smsyn.match
import smsyn.io.fits

def wav_exclude_to_wavmask(wav, wav_exclude):
    wavmask = np.zeros_like(wav).astype(bool) # Default: no points masked
    nwav_exclude = len(wav_exclude)
    for i in range(nwav_exclude):
        wav_min, wav_max = wav_exclude[i]
        wavmask[(wav_min < wav) & (wav < wav_max)] = True
    return wavmask

def grid_search(spec, libfile, wav_exclude, param_table, idx_coarse, idx_fine):
    """
    Args:
        segment (list): upper and lower bounds of segment
        libfile (str): path to library hdf5 file. 
        wav_exclude (list): define wavlengths to exclude from fit
            e.g. [[5018, 5019.5],[5027.5, 5028.5]] 
    """
    wavlim = spec.wav[0],spec.wav[-1]
    lib = smsyn.library.read_hdf(libfile,wavlim=wavlim)
    wavmask = wav_exclude_to_wavmask(spec.wav, wav_exclude)
    match = smsyn.match.Match(spec, lib, wavmask, cont_method='spline-dd')
    
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
        params[key].vary=False

    nodes = smsyn.match.spline_nodes(match.spec.wav[0],match.spec.wav[-1])
    smsyn.match.add_spline_nodes(params, nodes, vary=False)
    params['vsini'].vary = True

    print_grid_search()
    counter=0
    for i, row in param_table.iterrows():
        for key in param_keys:
            params[key].value = row[key]

        mini = lmfit.minimize(
            match.nresid, params, maxfev=100, method='leastsq'
        )
        #        mini = lmfit.minimize(match.nresid, params, method='brent')

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


def lincomb(spec, libfile, wav_exclude, param_table):
    """
    """
    ntop = len(param_table)
    wavlim = spec.wav[0],spec.wav[-1]
    lib = smsyn.library.read_hdf(libfile,wavlim=wavlim)
    wavmask = wav_exclude_to_wavmask(spec.wav, wav_exclude)
    match = smsyn.match.Match(spec, lib, wavmask, cont_method='spline-dd')
    model_indecies = np.array(param_table.model_index.astype(int))
    match = smsyn.match.MatchLincomb(
        spec, lib, wavmask, model_indecies, cont_method='spline-dd'
    )
    params = lmfit.Parameters()
    nodes = smsyn.match.spline_nodes(
        spec.wav[0],spec.wav[-1], angstroms_per_node=10
    )
    smsyn.match.add_spline_nodes(params, nodes, vary=False)
    smsyn.match.add_model_weights(params, ntop, min=0.01)
    params.add('vsini',value=5)
    params.add('psf',value=1.0, vary=False)

    def rchisq(params):
        nresid = match.masked_nresid(params)
        return np.sum(nresid**2) / len(nresid)

    out = lmfit.minimize(match.masked_nresid, params)
    nresid = match.masked_nresid(params)

    #print lmfit.fit_report(out)
    mw = smsyn.match.get_model_weights(out.params)
    mw = np.array(mw)
    mw /= mw.sum()

    params_out = lib.model_table.iloc[model_indecies]
    params_out = params_out['teff logg fe'.split()]
    params_out = pd.DataFrame((params_out.T * mw).T.sum()).T
    params_out['vsini'] = out.params['vsini'].value
    params_out['psf'] = out.params['psf'].value
    params_out['rchisq0'] = rchisq(params)
    params_out['rchisq1'] = rchisq(out.params)
    print params_out
    return params_out

def polish(matchlist, params0, angstrom_per_node=20, objective_method='chi2med'):
    """Polish parameters
    
    Given a list of match object, polish the parameters segment by segment

    Args:
        matchlist (list of smsyn.match.Match objects): list of Match objects.
            One for each wavelength segment to be fit
        params0 (lmfit.Parameters): lmfit.Parameters object with initial guesses
        angstrom_per_node (float): approximate separation between continuum and
            spline nodes. Number of nodes will be rounded to nearest integer.
        objective_method (string): name of objective function. Must be a method
            of the Match object.

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
