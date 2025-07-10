from util import *

pickle.load(cov_c, open("priors/%s/cov_c%s_%s_%s.p" % (sector, sector, cam, ccd), "wb" ) )
pickle.dump(cov_c2, open("priors/%s/cov_c_diag%s_%s_%s.p" % (sector, sector, cam, ccd), "wb") )
