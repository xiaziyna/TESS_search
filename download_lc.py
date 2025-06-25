import lightkurve as lk
import pickle
import numpy as np
from info import sectors

lc_data = {}
processed_lc_data = {}
time_data = {}
cam_data = {}
ccd_data = {}

tid_list = np.loadtxt('persistant_tids.txt', dtype=int)

for tid in tid_list:
    tic_id = str(tid)
    for sector in range(sectors[0], sectors[-1]+1):
        print (sector, tic_id)
        try:
            search_result = lk.search_lightcurve('TIC '+tic_id, mission='TESS', author='SPOC', cadence='short', sector=sector)  
            lightcurve = search_result.download_all()[0]
        except: 
            print ('missing sector', tic_id)
            continue
        lc_sap = lightcurve.SAP_FLUX # SAP = Simple Aperture Photometry = unprocessed data
        lc_pdc = lightcurve.PDCSAP_FLUX
        quality = lc_sap.quality # Quality, tells you when the lightcurve values are bad and should be masked 
        lc_data[sector] = lc_sap.flux.to_value()[~quality.astype(bool)]
        processed_lc_data[sector] = lc_pdc.flux.to_value()[~quality.astype(bool)]
        time_data[sector] = lc_sap.cadenceno[~quality.astype(bool)]
        cam_data[sector] = lightcurve.camera
        ccd_data[sector] = lightcurve.ccd
    pickle.dump((lc_data, processed_lc_data, time_data, cam_data, ccd_data), open('~/TESS/data/%s.p' % (tic_id), 'wb'))
    
