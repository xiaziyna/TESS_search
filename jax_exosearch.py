import lightkurve as lk
import numpy as np
import jax
import jax.numpy as jnp
from util import *
import scipy

def transit_num(y_d, num_period):
    ''''
    Arg:
    y_d : y.Cov_inv * transit_profile_d the size of this is ~ N_full / delta

    Returns:
    num_det: returns numerator of likelihoods as a 2D array indexed by period/delta and epoch/delta
    '''
    num_det = jnp.zeros((num_period, num_period)) 
    for p in range(num_period):
        for t0 in range(p+1):
            num_det = num_det.at[p, t0].set(jnp.sum(y_d[t0 :: p+1]))
    return num_det

def transit_den(K_d, num_period):
    '''
    Arg:
    K_d : Cov_inv * (transit_profile_d.transit_profile_d^T) the size of this is ~ (N_full / delta, N_full / delta)

    Returns:
    den_det: returns denominator of likelihoods as a 2D array indexed by period/delta and epoch/delta
    '''
    den_det = jnp.zeros((num_period, num_period))
    for p in range(num_period):
        for t0 in range(p+1): 
            den_det = den_det.at[p, t0].set(jnp.sum(K_d[t0 :: p+1, t0 :: p+1]))
    return den_det


# Load lc and inverse covariance model
# ====================================

#(lc_data, processed_lc_data, detrend_data, norm_offset, quality_data, time_data, cam_data, ccd_data, coeff_ls, centroid_xy_data, pos_xy_corr) = pickle.load(open(os.path.expanduser('~/TESS/data/%s.p' % (tic_id)), 'rb')) 
tid = ...
sector = 73..
lc_detrend, cov_inv = covariance_sector(tid, sector)

# ====================================
# Defining transit parameter search space (period, epoch, duration)
# Period ranges from 0 to N/2
# epoch ranges from 0 to P

delta = 5 # period and epoch step size in 2-minute samples
durations = jnp.array([1, 2, 3, 4, 6, 8, 10, 12, 14, 16])*30 
N_full = len(lc_detrend)
lc_cov_inv = cov_inv.dot(lc_detrend) # y^T Cov_inv
# Compute transit likelihoods over parameter space 

num_period = int((N_full - durations[-1]) // (2 * delta))# number of periods to search in stepsize of delta
transit_likelihood_stats = np.zeros((len(durations), num_period, num_period))

jit_lik_num = jax.jit(transit_num, static_argnums=(1))
jit_lik_den = jax.jit(transit_den, static_argnums=(1))

for i in range(len(durations)):
    print (i)
    if i != 5: continue # only compute the LRT for one of the trial transit durations
    d = durations[i]
    transit_profile = jnp.ones(d)
    transit_kernel = jnp.outer(transit_profile, transit_profile)

    y_d = jax.scipy.signal.convolve(lc_cov_inv, transit_profile)[int(d/2)-1:N_full-int(d/2)-1][::delta]

    # commented this out as I get a memory error
    #K_d = jax.scipy.signal.convolve2d(cov_inv, transit_kernel)[int(d/2)-1:N_full-int(d/2)-1,int(d/2)-1:N_full-int(d/2)-1][::delta,::delta]

    # different way to calculate K_d
    K_d = np.zeros((np.shape(y_d)[0], np.shape(y_d)[0]))
    for l in range(num_period):
        for m in range(num_period):
            K_d[l,m] = np.sum(transit_kernel*cov_inv[(l*delta):(l*delta) + d, (m*delta):(m*delta) + d])    
    K_d = jnp.array(K_d)

    likelihoods_num = transit_num(y_d, num_period)
    likelihoods_den = transit_den(K_d, num_period)

    # Output transit detection tests, indexed as [P/delta, t_0/delta]
    transit_likelihood_stats[i] = np.divide(likelihoods_num, np.sqrt(likelihoods_den), out=np.zeros_like(likelihoods_num), where=likelihoods_den!=0.)
    
top_detections = nd_argsort(transit_likelihood_stats)

for i in range(5):
    print (top_detections[i], transit_likelihood_stats[top_detections[i][0], top_detections[i][1], top_detections[i][2]])
    print ('LRT (SNR): ', np.round(transit_likelihood_stats[top_detections[i][0], top_detections[i][1], top_detections[i][2]],2), 'duration (hr): ', np.round(top_detections[i][0]/30, 2), 'period (day): ', np.round(delta*(top_detections[i][1]+1)/(day_to_cadence),2), 'epoch(day): ',  np.round((delta/day_to_cadence)*top_detections[i][2], 2) )
