import h5py
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from scipy.stats import chi2
from astropy.cosmology import FlatLambdaCDM
from astropy import constants as consts
from astropy import units
import pandas as pd
from matplotlib.colors import LogNorm
import IPython

import os, sys
import numpy as np
import matplotlib.pyplot as plt 
from drift.core.manager import ProductManager
import IPython
from IPython import get_ipython

label='dk_1thresh_fg_10thresh_updated_256'
name='psmc_'+label

#edit='_mod.txt'
#cmd='grep -v None ps_psmc_$label.txt > ps_psmc_$label$edit'
#os.system('grep -v None ps_psmc_$label.txt > ps_psmc_$label$edit')

kc, pk_cora, pk_est = np.loadtxt('ps_psmc_dk_1thresh_fg_10thresh_updated_256_mod.txt',unpack=True)
kcenter, pa, err = np.loadtxt('ps_err_dk_1thresh_fg_10thresh_updated_256.txt',unpack=True)

mask=np.zeros(len(kcenter), dtype = bool) 
for kci in kc: 
    mask[kci == kcenter]=True

errs=err[mask]
delpk=pa[mask]
#print(kcenter[mask] - kc)
#print(len(errs))

#IPython.embed()
pk=pk_est*(1.+delpk/pk_est)
errs=errs*(1.+delpk)
j=np.argsort(kc)

ax = plt.axes()    
#ax.set_xscale("log")   
#ax.set_yscale("log")  
ax.loglog(kc[j],pk_cora[j], label='Fiducial P(k) (Cora)')
#ax.plot(kc[j],2.*pk[j], marker='.', label='Estimated P(k)')
ax.errorbar(kc[j],pk[j], 5.*errs[j], label='Estimated P(k)') #, fmt='.')
plt.legend() 
plt.xlabel('$k_{center}$')
#plt.xlim(0.2,0.3)
plt.ylabel('P(k)')
plt.title(name)
plt.savefig('av_pks_'+name+'.png')
plt.show() 

#IPython.embed()
