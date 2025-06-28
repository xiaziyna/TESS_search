import lightkurve as lk
import pickle
import numpy as np
from util import smooth_p
from info import *

(_, _, detrend_data, norm_offset, quality, time_data, _, _, _) = pickle.load(open('~/TESS/data/%s.p' % (tic_id), 'rb'))

for sector in sectors:
    N_cadence = cadence_bounds[sector][1]-cadence_bounds[sector][0]
    filled_lc = np.zeros(N_cadence)
    cadence = time_data[sector] - cadence_bounds[sector][0]
    filled_lc[cadence] = threshold_data(detrend_data[sector], level=3)
    std_ = np.std(filled_lc[time_data[sector] - cadence_bounds[sector][0]])
    filled_lc[~cadence] = np.random.normal(0, std_, N_cadence - len(cadence))
    zp_lc = np.zeros((2*N_cadence) - 1)
    zp_lc[:N_cadence] = detrend
    p_noise_smooth = smooth_p(zp_lc, K=3)
