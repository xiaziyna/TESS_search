import os
import numpy as np
from info import cadence_bounds, sectors
from sklearn.covariance import empirical_covariance
from scipy.stats import norm
import lightkurve as lk

"""
put quick_lc_dl in util python script maybe
"""
def quick_lc_dl(sector, tic_id):
    try:
        search_result = lk.search_lightcurve('TIC '+tic_id, mission='TESS', author='SPOC', cadence='short', sector=sector)  
        lightcurve = search_result.download_all()[0]
    except: 
        print ('missing sector', tic_id)
        return
    lc_sap = lightcurve.SAP_FLUX # SAP = Simple Aperture Photometry = unprocessed data
    quality = lc_sap.quality # Quality, tells you when the lightcurve values are bad and should be masked 
    lc_data = lc_sap.flux.to_value()[~quality.astype(bool)]
    cadence_data = lc_sap.cadenceno[~quality.astype(bool)]
    return lc_data, cadence_data

def gen_cov_c(sector, cam, ccd):
    evecs = pickle.load(open("evec_matrix_%s_%s_%s.p" % (sector,cam, ccd), "rb" ))
    N_vec = evecs.shape[0]
    tid_list = []
    with open(f'cam{cam}_ccd{ccd}_tids.txt') as f:
        for line in f:
            first, second = line.strip().split('\t')
            ids = [int(x) for x in second.split(',')]
            if sector in ids:
                tid_list.append(str(first))
    N = len(tid_list)
    print (N)
    coeff_ls = np.zeros((N, N_vec))
    for lc_i in range(N):
        tid = str(tid_list[lc_i])
        try:
            lc, cadence_data = quick_lc_dl(sector, tid)
        except: continue
        cadence_data -= cadence_bounds[sector][0]
        cadence_data -= 1
        lc_offset = np.nanmedian(lc)
        lc -= lc_offset
        lc_norm = np.linalg.norm(lc)
        lc /= lc_norm
        evecs_mask = evecs[:, cadence_data]
        coeff = np.dot(evecs_mask, lc.T)
        coeff_ls[lc_i] = coeff
    return coeff_ls

for sector in range(sectors[0], sectors[1]+1):
    for i in range(4):
        for j in range(4):
            print (i, j)
            coeff_ls = gen_cov_c(sector, i, j)
            print ('nan check',  np.sum(np.isnan(coeff_ls)))
            coeff_ls_center = coeff_ls - np.mean(coeff_ls, axis=0, keepdims=True)
            cov_c = empirical_covariance(coeff_ls_center)
            cov_c2 = np.diag(np.var(coeff_ls_center, axis=0).astype('float32'))
            pickle.dump(coeff_ls, open("priors/%s/coeff_ls%s_%s_%s.p" % (sector, sector, cam, ccd), "wb" ) )
            pickle.dump(cov_c, open("priors/%s/cov_c%s_%s_%s.p" % (sector, sector, cam, ccd), "wb" ) )
            pickle.dump(cov_c2, open("priors/%s/cov_c_diag%s_%s_%s.p" % (sector, sector, cam, ccd), "wb") )

c_ind = 1
fig, ax = plt.subplots(1, 1)
mean = np.mean(coeff_ls[:,c_ind])
var = np.var(coeff_ls[:,c_ind])
x = np.linspace(-200, 200, 100)
ax.plot(x, norm.pdf(x, mean, np.sqrt(var)), label = 'Coefficient Prior', color = '#C95000', linewidth=2.5)#plt.plot(5000*norm.pdf(x_axis,mean,var))
ax.hist(coeff_ls[:,c_ind], density=True, cumulative = False, histtype='stepfilled', color = '#FF8010', alpha=0.5, bins = 200, label = 'LS Coefficient Values')
ax.set_xlim(-200,200)
ax.set_ylabel('Density', fontsize = 8)
ax.legend(frameon=False, fontsize=6, loc=2)
ax.set_xlabel('Coefficient Amplitude', fontsize = 8)
ax.set_title('Empirical Cumulative Coefficient Prior, Leading basis term, Sector %s,  Cam %s, CCD %s' % (sector, cam, ccd), fontsize = 10)
plt.show()
#plt.savefig(dir+'images/other/c_coeff_prior%s_%s_%s.png' % (sector, cam, ccd))
