import lightkurve as lk

def get_cbv_cadence_bounds(tic_id, sectors=range(70, 84)):
    bounds = {}

    for sector in sectors:
        try:
            # Download one lightcurve for this TIC in this sector
            lc = lk.search_lightcurve(f"TIC {tic_id}").download()

            # Extract camera and CCD from the lightcurve
            cbvs = lk.correctors.load_tess_cbvs(
                sector=sector,
                camera=lc.camera,
                ccd=lc.ccd,
                cbv_type="SingleScale"
            )

            cstart = int(cbvs.cadenceno[0])
            cend = int(cbvs.cadenceno[-1])
            bounds[sector] = (cstart, cend)

        except Exception as e:
            print(f"✘ Sector {sector} failed: {e}")
            bounds[sector] = None

    return bounds

# Example usage:
cadence_bounds = get_cbv_cadence_bounds(tic_id=2733409)
print (cadence_bounds)
