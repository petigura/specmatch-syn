"""Top-level fitting and polishing routines

"""
import numpy as np
import lmfit

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

    
    param_table = param_table0.copy()
    for col in 'chisq rchisq logprob niter'.split():
        param_table[col] = np.nan

    params = lmfit.Parameters()
    for key in 'teff logg fe vsini psf'.split():
        params.add(key)
        params[key].vary = False

    params['vsini'].vary = True
    for i, row in param_table.iterows():
        
        mini = lmfit.minimize(match.nresid)
        
        
        mini.params
        
