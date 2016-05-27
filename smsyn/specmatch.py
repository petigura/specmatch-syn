"""Top-level fitting and polishing routines

"""
import numpy as np
import lmfit
import pandas as pd
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



def polish(match, params0):
    """Polish parameters
    
    Given a match object, polish the parameters
    """

    lmfit.minimize(match.chi2med,params, method='nelder')
