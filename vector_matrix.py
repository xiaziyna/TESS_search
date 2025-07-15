from astropy.io import fits
import numpy as np
from info import sectors, cadence_bounds
import glob
# Format cotrending basis vectors into a full-size matrix, between the beginning and end cadence of each sector

for sector in range(sectors[0], sectors[1]+1)
    for cam in range(1, 5):
        for ccd in range(1, 5):
            N_vecs = 8
            N_cadence = cadence_bounds[sector][1] - cadence_bounds[sector][0]
            evecs = np.zeros((N_vecs, N_cadence), dtype='float32')
            
            pattern = f'TESS/data/info/light_curves/cbvs/*{sector}-{cam}-{ccd}*-cbv.fits'
            file_path = glob.glob(pattern)[0]
            with fits.open(file_path, memmap=True) as hdulist:
                print(f"Loaded {file_path}")
                for j in range(30):
                    k = j+1
                    try:
                        evec = hdulist2[1].data['VECTOR_%s' % k]
                        times = hdulist2[1].data['CADENCENO']
                        if np.any(evec): N_vecs += 1
                        evecs[j, times - cadence_bounds[sector][0]] = evec
                    except: continue
            pickle.dump(evecs, open( s_dir+"priors/%s/evec_matrix_%s_%s_%s.p" % (sector, sector, cam, ccd), "wb" ) ) #change path
