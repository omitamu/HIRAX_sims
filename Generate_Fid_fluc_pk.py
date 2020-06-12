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

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

from cora.signal import corr21cm
cr = corr21cm.Corr21cm()
from drift.core.psestimation import decorrelate_ps_file

def fg_wedge(kperp, z=1.33, theta=5*units.deg):
    return theta.to('rad').value*(cosmo.comoving_distance(z)*cosmo.H(z)/consts.c/(1+z)).to('').value*kperp

def plot_power_spectrum(measured_fname, prod_fname, label=None):
    
    with h5py.File(measured_fname, 'r') as fil:
        pa = fil['powerspectrum'][()]
        err=fil['fisher'][()]
    with h5py.File(prod_fname, 'r') as fil:
        kpar = fil['kpar_center'][()]
        kpar_bands = fil['kpar_bands'][()]
        kperp = fil['kperp_center'][()]
        kperp_bands = fil['kperp_bands'][()]
        cov = fil['covariance'][()]
        fish = fil['fisher'][()]
        
    kperp_size = len(kperp_bands) - 1
    kpar_size = len(kpar_bands) - 1
    
    kperp = kperp.reshape((kperp_size, kpar_size)).T
    kpar = kpar.reshape((kperp_size, kpar_size)).T
    
    k_center = (kpar**2 + kperp**2)**0.5
    
    pa = pa.reshape((kperp_size, kpar_size)).T
    errs = np.sqrt(np.diag(cov)).reshape((kperp_size, kpar_size)).T
    print(errs.shape) #np.sqrt(np.diag(cov))
    fig, (cov_ax, amp_ax, err_ax) = plt.subplots(1, 3, figsize=(20, 6), sharey=True)

    extent = (kperp[0, 0], kperp[0, -1], kpar[0, 0], kpar[-1, 0])
     
    # Plot amplitudes
    im = amp_ax.imshow(pa, origin='lower', extent=extent, cmap='RdBu_r', vmin=0, vmax=1) #0.5)#1e11)
    plt.colorbar(im, ax=amp_ax, label='PS Relative Amplitudes')

    amp_ax.set_xlabel('$k_{\perp}$ [h Mpc$^{-1}$]')
    amp_ax.set_ylabel('$k_{\parallel}$ [h Mpc$^{-1}$]')
    
    # Plot diagonal uncertainties
    im = err_ax.imshow(errs, origin='lower', extent=extent, cmap='RdBu_r', vmin=0, vmax=0.1)
    plt.colorbar(im, ax=err_ax, label='PS Relative Err')

    err_ax.set_xlabel('$k_{\perp}$ [h Mpc$^{-1}$]')
    err_ax.set_ylabel('$k_{\parallel}$ [h Mpc$^{-1}$]')

    fig.suptitle(label)
    
    # Approx. BAO scales
    inner = plt.Circle((0, 0), 0.03, color='k', fc='none', lw=3, ls='--')
    outer = plt.Circle((0, 0), 0.2, color='k', fc='none', lw=3, ls='--')
    ax = plt.gca()
    ax.add_artist(inner)
    ax.add_artist(outer)

    # Plot covariance matrix
    #err=err.reshape((kperp_size, kpar_size)).T
    df=pd.DataFrame(err)
    #plt.matshow(df, ax=cov_ax, label='Covariance')
    im = cov_ax.imshow(df, extent=extent, cmap='RdBu_r',vmin=0,vmax=10)
    plt.colorbar(im, ax=cov_ax, label='Covariance')
        
    return fig, im

####################################################################################################3

prod_dir='drift_prod_hirax_survey_9elem_3point_24bands/bt/'
out_dir='drift_ts_hirax_survey_9elem_3point_24bands/'

label='dk_1thresh_fg_10thresh_updated_256'
#'dk_5thresh_fg_10thresh_updated_256'
#'dk_5thresh_fg_100thresh_updated'
#'dk_1thresh_fg_10thresh_updated_256'
#'dk_0thresh_fg_0thresh_updated'
#'kl_0thresh_fg_0thresh_256'

#prod_fname=prod_dir+label+'/psmc_'+label+'/fisher.hdf5'
#measured_fname=out_dir+'ps_psmc_'+label+'.hdf5'

prod_fname='../psmc_'+label+'_fisher.hdf5'
measured_fname='../ps_psmc_'+label+'.hdf5'

with h5py.File(prod_fname, 'r') as fil:
        kpar = fil['kpar_center'][()]
        kpar_bands = fil['kpar_bands'][()]
        kperp = fil['kperp_center'][()]
        kperp_bands = fil['kperp_bands'][()]
        cov = fil['covariance'][()]
        fish = fil['fisher'][()]
      
with h5py.File(measured_fname, 'r') as fil:
        pa = fil['powerspectrum'][()]
        err=fil['fisher'][()]      
        
kperp_size = len(kperp_bands) - 1; kpar_size = len(kpar_bands) - 1; 
#kperp = kperp.reshape((kperp_size, kpar_size)).T
#kpar = kpar.reshape((kperp_size, kpar_size)).T
k_center = (kpar**2 + kperp**2)**0.5

#pa = pa.reshape((kperp_size, kpar_size)).T
#errs = np.sqrt(np.diag(cov)) #.reshape((kperp_size, kpar_size)).T
errs = np.sqrt(np.diag(cov))

IPython.embed()

err_file=open('ps_err_'+label+'.txt','w')
for i in range(len(k_center)): 
    print(k_center[i], pa[i], errs[i], file=err_file) 

#P(k) measured fluctuations    
plt.clf()    
fig, im = plot_power_spectrum(prod_fname=prod_fname,measured_fname=measured_fname,label=label+' 5\% Gain Unc.')
plt.savefig(label+'.png')
plt.show()

sys.exit()

manager = ProductManager.from_config('prod_params.yaml')
name='psmc_'+label

ps_est = manager.psestimators[name] #'psmc_kl_0thresh_fg_0thresh_128']
ps_est.genbands()


from scipy.integrate import dblquad
from multiprocessing import Pool

pool = Pool(processes=8)
av_pk=[]

def perform_integral(i):
    k_range = np.sqrt(ps_est.kpar_start[i]**2 + ps_est.kperp_start[i]**2), np.sqrt(ps_est.kpar_end[i]**2 + ps_est.kperp_end[i]**2)
    integrated = dblquad(ps_est.band_pk[i], 0, 1, lambda _: k_range[0], lambda _: k_range[1])[0]
    weight = dblquad(ps_est.band_func[i], 0, 1, lambda _: k_range[0], lambda _: k_range[1])[0]
    #print(k_range,integrated,weight)
    if weight == 0:
        print('Weight zero',integrated, weight) 
    else:     
        print('Nonzero', integrated, weight) 
        av_pk.append(integrated/weight) 
        return integrated/weight 
                                        
#av_pks = pool.
av_pks = map(perform_integral, range(len(ps_est.band_pk)))
print('integral done')

from cora.signal import corr21cm
cr = corr21cm.Corr21cm()

# Compare with power spectrum evaluated at k bin center
filename=open('ps_'+name+'.txt','w')
# Compare with power spectrum evaluated at k bin center
for i, pk in enumerate(av_pks):
    #print >> filename, ps_est.k_center[i],pk, cr.ps_vv(ps_est.k_center[i])
    print(i,len(ps_est.k_center)) 

j=np.argsort(ps_est.k_center)
kc=ps_est.k_center[j]
pk=np.array(av_pk)
#IPython.embed()
pk_est=pk[j]
pk_cora=cr.ps_vv(kc)
for i in range(len(kc)): 
    print(kc[i], pk_cora[i], pk_est[i], errs[i], file=filename) 
filename.close()

ax = plt.axes()    
ax.set_xscale("log")   
ax.set_yscale("log")  
ax.loglog(kc,pk_cora, label='Fiducial P(k) (Cora)')  
ax.errorbar(kc,pk_est, 10*errs, label='Estimated P(k)') 
plt.legend() 
plt.xlabel('$k_{center}$')
plt.ylabel('P(k)')
plt.title(name)
plt.savefig('av_pks_'+name+'.png')
plt.show() 
    
#P(k) measured fluctuations    
plt.clf()    
fig, im = plot_power_spectrum(prod_fname=prod_fname,measured_fname=measured_fname,label=label+' 5\% Gain Unc.')
plt.savefig(label+'.png')
plt.show()
    
