import lightkurve as lk
import pickle
import numpy as np
from util import smooth_p
from info import *

(_, _, detrend_data, norm_offset, quality, time_data, _, _, _) = pickle.load(open('~/TESS/data/%s.p' % (tic_id), 'rb'))

for sector in sectors:
    cadence_data = time_data[sector] - cadence_bounds[sector][0] -1
    N_cadence = cadence_bounds[sector][1]-cadence_bounds[sector][0]
    filled_lc = np.zeros(N_cadence)
    mask = np.zeros(N_cadence, dtype=bool)
    mask[cadence_data] = True
    #sim_exo = box_transit(cadence_data, 720*6, 30*5, 48, 0.01*lc_norm)
    #detrend_data[sector] += sim_exo

    filled_lc[mask] = threshold_data(detrend_data[sector], level=3)
    std_ = np.std(filled_lc[cadence_data])
    filled_lc[~mask] = np.random.normal(0, std_, N_cadence - len(cadence_data))

    zp_lc = np.zeros((2*N_cadence) - 1)
    zp_lc[:N_cadence] = filled_lc
    p_noise_smooth = smooth_p(zp_lc, K=3)
    ac = np.real(np.fft.ifft(p_noise_smooth)).astype('float32')
    ac = ac[:N_cadence]
