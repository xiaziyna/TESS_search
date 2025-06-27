import lightkurve as lk
import pickle
import numpy as np
from info import sectors, cadence_bounds
import os


tid_list = np.loadtxt('persistant_tids.txt', dtype=int)

for tid in tid_list:
    tic_id = str(tid)
    if os.path.exists(os.path.expanduser(f'~/TESS/data/{tic_id}.p')): 
        continue
    else:
        raise SystemExit
        lc_data = {}
        processed_lc_data = {}
        time_data = {}
        cam_data = {}
        ccd_data = {}
        centroid_xy_data = {}
        coeff_ls = {}
        detrend_data = {}
        norm_offset = {}
        quality = {}
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
            quality[sector] = lightcurve.quality
            centroid_xy_data[sector] = [lightcurve.centroid_col, lightcurve.centroid_row] #are these the right way around?
            evecs = pickle.load(open("priors/%s/evec_matrix_%s_%s_%s.p" % (sector, sector, cam_data[sector], ccd_data[sector]), "rb" ) #add correct path!
            cadence_data = time_data[sector] - cadence_bounds[sector][0]
            lc_offset = np.nanmedian(lc_sap)
            lc_sap -= lc_offset
            lc_norm = np.linalg.norm(lc_sap)
            lc_sap /= lc_norm
            evecs_mask = evecs[:, cadence_data]
            coeff = np.dot(evecs_mask, lc.T)
            coeff_ls[sector] = coeff
            norm_offset[sector] = [lc_offset, lc_norm]
            coeff_tid[sector] = coeff*lc_norm
            detrend_data[sector] = lc_norm*(lc_sap - np.dot(coeff, evecs)) 
            # uncomment me on first run, then delete - just to check the output looks fine!
            # import matplotlib.pyplot as plt
            # plt.figure
            # plt.scatter(cadence_data, detrend[sector])
            # plt.show
        pickle.dump((lc_data, processed_lc_data, detrend_data, norm_offset, quality, time_data, cam_data, ccd_data, coeff_ls), open('~/TESS/data/%s.p' % (tic_id), 'wb'))
