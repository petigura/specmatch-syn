from astropy.io import ascii
import pandas as pd
import numpy as np
tab1 = ascii.read('J_ApJ_767_127/table1.dat',readme='J_ApJ_767_127/ReadMe')
tab2 = ascii.read('J_ApJ_767_127/table2.dat',readme='J_ApJ_767_127/ReadMe')
tab1 = pd.DataFrame(np.array(tab1))
tab2 = pd.DataFrame(np.array(tab2))

from astropy import constants as c

tab1 = tab1.rename(
    columns={
        'KOI':'koi',
        'KIC':'kic',
        'e_numax':'unumax',
    }
)
tab2 = tab2.rename(
    columns={
        'KOI':'koi',
        'KIC':'kic',
        'Teff':'teff',
        'e_Teff':'uteff',
        '[Fe/H]':'fe',
        'e_[Fe/H]':'ufe',
        'Mass':'mass',
        'e_Mass':'umass',
        'Rad':'rad',
        'e_Rad':'urad',
    }
)

tab = pd.merge(tab1,tab2,on=['koi','kic'])    

#g_sun = c.G * c.M_sun * c.R_sun**-2
#teff_sun = 5770
#numax_sun = 3090
#g = g_sun * (tab.teff / teff_sun)**0.5 * (tab.numax / numax_sun)
#ug = g * np.sqrt( (0.5*tab.uteff / tab.teff)**2 + (tab.unumax / tab.numax) )

g = c.G * (np.array(tab.mass) * c.M_sun) * (np.array(tab.rad) * c.R_sun )**-2
ug = g * np.sqrt( (tab.umass/ tab.mass)**2 + (2.0 * tab.urad / tab.rad)**2 )
logg = np.log10( g.cgs.value )
ulogg = np.log10(g.cgs.value + ug.cgs.value) - logg
tab['logg'] = logg
tab['ulogg'] = ulogg
tab['koi kic teff uteff logg ulogg fe ufe'.split()].to_csv('huber13.csv')
