import numpy as np
from numpy.polynomial.polynomial import Polynomial

# Identify gap indices (likely downlinks)
gap_threshold = 360 #1/2 day
dt = np.diff(cadence_data)
gap_indices = np.where(dt > gap_threshold)[0]

# Include start and end for segmentation
split_indices = np.concatenate(([0], gap_indices + 1, [len(time)]))

# Detrend each segment
flux_detrended = np.zeros_like(flux)
for i in range(len(split_indices) - 1):
    start, end = split_indices[i], split_indices[i+1]
    start = np.max(split_indices[i], split_indices[i+1] - 360)
    t_seg = time[start:end]
    f_seg = flux[start:end]

    # Remove NaNs
    valid = np.isfinite(t_seg) & np.isfinite(f_seg)
    if np.count_nonzero(valid) < 5:
        flux_detrended[start:end] = f_seg  # too few points to fit
        continue

    t_valid = t_seg[valid]
    f_valid = f_seg[valid]

    # Fit polynomial (degree can be adjusted)
    coeffs = Polynomial.fit(t_valid, f_valid, deg=2).convert().coef
    trend = Polynomial(coeffs)(t_seg)
    flux_detrended[start:end] = f_seg - trend

# Replace lc.flux with detrended result
lc_detrended = lc.copy()
lc_detrended.flux = flux_detrended

# Optionally: plot
lc_detrended.plot()
