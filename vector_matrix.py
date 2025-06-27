from astropy.io import fits
import numpy as np
from info import sectors, cadence_bounds

# Format cotrending basis vectors into a full-size matrix, between the beginning and end cadence of each sector

for sector in range(sectors[0], sectors[1]+1)
    for cam in range(1, 5):
        for ccd in range(1, 5):
            N_vecs = 0 #may only be 8 cbv
            #N_cadence = cadence_bounds[sector][1] - cadence_bounds[sector][0]
            #evecs = np.zeros((N_vecs, N_cadence), dtype='float32')
            with fits.open('%s %s %s' % (sector, cam, ccd), memmap=True) as hdulist2: #change path/filename
                for j in range(30):
                    k = j+1
                    try:
                        evec = hdulist2[1].data['VECTOR_%s' % k]
                        times = hdulist2[1].data['CADENCENO']
                        if np.any(evec): N_vecs += 1
                        #evecs[j, times - cadence_bounds[sector][0]] = evec
                    except: continue
                #pickle.dump(evecs, open( s_dir+"priors/%s/evec_matrix_%s_%s_%s.p" % (sector, sector, cam, ccd), "wb" ) ) #change path
            print ('sector, cam, ccd: ', sector, cam, ccd, N_vecs)
