import os
from astropy.io import fits

def load_cbv(sector: int, camera: int, ccd: int, directory: str = "."):
    # Zero-pad sector
    sector_str = f"{sector:04d}"
    # Find filename pattern
    for fname in os.listdir(directory):
        if (
            f"-s{sector_str}-" in fname
            and f"{camera}-{ccd}" in fname
            and fname.endswith("_cbv.fits")
        ):
            filepath = os.path.join(directory, fname)
            print(f"Opening: {filepath}")
            with fits.open(filepath) as hdul:
                hdul.info()
                return hdul
    raise FileNotFoundError(f"No CBV FITS found for sector {sector}, cam {camera}, ccd {ccd}")

# Example usage:
hdul = load_cbv(sector=70, camera=4, ccd=4)
