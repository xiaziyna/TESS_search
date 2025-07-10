import lightkurve as lk
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
from info import sectors, cadence_bounds

tid_list = np.loadtxt('persistant_tids.txt', dtype=int)

for tid in tid_list:
    tic_id = str(tid)
    if os.path.exists(os.path.expanduser(f'~/TESS/data/{tic_id}.p')): 
        continue
    else:
        lc_data = {}
        processed_lc_data = {}
        time_data = {}
        cam_data = {}
        ccd_data = {}
        centroid_xy_data = {}
        coeff_ls = {}
        detrend_data = {}
        norm_offset = {}
        quality_data = {}
        pos_xy_corr = {}
        for sector in range(sectors[0], sectors[-1]+1):
            print (sector, tic_id)
            try:
                search_result = lk.search_lightcurve('TIC '+tic_id, mission='TESS', author='SPOC', cadence='short', sector=sector)  
                lightcurve = search_result.download_all()[0]
            except: 
                print ('missing sector', tic_id)
                continue
            lc_sap = lightcurve.SAP_FLUX
            lc_pdc = lightcurve.PDCSAP_FLUX
            quality = lc_sap.quality 
            lc_data[sector] = lc_sap.flux.to_value()[~quality.astype(bool)]
            processed_lc_data[sector] = lc_pdc.flux.to_value()[~quality.astype(bool)]
            time_data[sector] = lc_sap.cadenceno[~quality.astype(bool)]
            cam_data[sector] = lightcurve.camera
            ccd_data[sector] = lightcurve.ccd
            qual = np.zeros_like(lc_sap.cadenceno.data)
            qual[quality.astype(bool)] = 1
            quality_data[sector] = qual
            centroid_xy_data[sector] = [lc_sap.mom_centr1.to_value()[~quality.astype(bool)], lc_sap.mom_centr2.to_value()[~quality.astype(bool)]] 
            pos_xy_corr[sector] = [lc_sap.pos_corr1.to_value()[~quality.astype(bool)], lc_sap.pos_corr2.to_value()[~quality.astype(bool)]]
            evecs = pickle.load(open("priors/%s/evec_matrix_%s_%s_%s.p" % (sector, sector, cam_data[sector], ccd_data[sector]), "rb" ))
            cadence_data = time_data[sector] - cadence_bounds[sector][0] -1
            lc_sap_val = lc_sap.flux.to_value()[~quality.astype(bool)]
            lc_offset = np.nanmedian(lc_sap_val)
            lc_sap_val -= lc_offset
            lc_norm = np.linalg.norm(lc_sap_val)
            lc_sap_val /= lc_norm
            evecs_mask = evecs[:, cadence_data]
            coeff = np.dot(evecs_mask, lc_sap_val.T)
            norm_offset[sector] = [lc_offset, lc_norm]
            coeff_ls[sector] = coeff*lc_norm
            detrend_data[sector] = lc_norm*(lc_sap_val - np.dot(coeff, evecs_mask)) 
        pickle.dump((lc_data, processed_lc_data, detrend_data, norm_offset, quality_data, time_data, cam_data, ccd_data, coeff_ls, centroid_xy_data, pos_xy_corr),  open(os.path.expanduser('~/TESS/data/%s.p' % (tic_id)), 'wb')) 
