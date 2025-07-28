import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, chi2
import lightkurve as lk
import pickle
import os
from info import cadence_bounds

def even_odd_phase_our_data(lc, period, time, t0, plot_size=False, same_axes=False, binning=False, duration=None, plot=True):
    
    if binning != False:
        lc = lc.bin(3)
    time_data = time #in cadence right now
    flux = lc
    t0_odd = t0
    t0_even = t0 + period
    period2 = period * 2
    phase_odd = np.array([-0.5 + ((t - t0_odd - 0.5*period2) % period2) / period2 for t in time])
    phase_even = np.array([-0.5 + ((t - t0_even - 0.5*period2) % period2) / period2 for t in time])
    p_val = None
    odd_depths = even_depths = None
    if duration is not None:
        in_transit_odd = np.abs(phase_odd) < (duration / (2 * period))
        in_transit_even = np.abs(phase_even) < (duration / (2 * period))
        odd_flux = np.array(flux[in_transit_odd])
        even_flux = np.array(flux[in_transit_even])
        odd_depths = 1 - odd_flux
        even_depths = 1 - even_flux
        odd_clean = odd_depths[~np.isnan(odd_depths)]
        even_clean = even_depths[~np.isnan(even_depths)]
        if len(odd_clean) > 0 and len(even_clean) > 0:
            try:
                t_stat, p_val = ttest_ind(odd_clean, even_clean)
                mean_odd = np.nanmean(odd_depths)
                mean_even = np.nanmean(even_depths)
                if p_val < 0.05:
                    print('Statistically significant difference between odd and even transits')
            except Exception as e:
                print(f"Error in t-test: {e}")
                p_val = np.nan
                mean_odd = mean_even = np.nan
        else:
            p_val = np.nan
            mean_odd = mean_even = np.nan
    else:
        mean_odd = mean_even = None
    if plot:
        if not same_axes:
            fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
            ax[0].plot(phase_odd, flux, lw=0, color='navy', marker='.', alpha=0.4)
            ax[1].plot(phase_even, flux, lw=0, color='maroon', marker='.', alpha=0.4)
            ax[0].set_xlabel("Phase")
            ax[0].set_ylabel("Normalized flux")
            ax[1].set_xlabel("Phase")
            ax[0].annotate("ODD", (0.3, np.nanmin(flux)), fontsize=14)
            ax[1].annotate("EVEN", (0.3, np.nanmin(flux)), fontsize=14)
            plt.subplots_adjust(wspace=0.02)
            if plot_size != False:
                ax[0].set_xlim(-plot_size, plot_size)
                ax[1].set_xlim(-plot_size, plot_size)
            if duration is not None and p_val is not None:
                stat_text = f"Odd mean depth: {mean_odd:.4g}\nEven mean depth: {mean_even:.4g}\np-value: {p_val:.2g}"
                ax[0].text(0.05, 0.95, stat_text, transform=ax[0].transAxes, fontsize=10, va='top', bbox=dict(facecolor='white', alpha=0.7))
        else:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(phase_odd, flux, lw=0, color='navy', marker='.', alpha=0.4, label='odd')
            ax.plot(phase_even, flux, lw=0, color='maroon', marker='.', alpha=0.4, label='even')
            ax.set_xlabel("Phase")
            ax.set_ylabel("Normalized flux")
            plt.legend()
            if plot_size != False:
                ax.set_xlim(-plot_size, plot_size)
            if duration is not None and p_val is not None:
                stat_text = f"Odd mean depth: {mean_odd:.4g}\nEven mean depth: {mean_even:.4g}\np-value: {p_val:.2g}"
                ax.text(0.05, 0.95, stat_text, transform=ax.transAxes, fontsize=10, va='top', bbox=dict(facecolor='white', alpha=0.7))
        plt.tight_layout()
        plt.show()
    return p_val


tjd_start_dict = {73: 3445.567, 74: 3470.657, 75:3494.894, 76:3519.372 , 77:3545.051 , 78:3569.402 , 79:3594.317 , 80:3621.724 , 81:3644.990 , 82:3667.042 , 83:3690.635} #start time of each sector in Tess Julian Date
even_odd_test_results = []

plot = False 

for tid in tic_ids[500:505]:
    (lc_data, processed_lc_data, detrend_data, norm_offset, quality_data, time_data, cam_data, ccd_data, coeff_ls, centroid_xy_data, pos_xy_corr) = pickle.load(open(os.path.expanduser('TESS/data/light_curves/%s.p' % (tid)), 'rb'))
    sector_pvals = []
    for sector in range(73, 84):
        lc_sector = lc_data[sector].unmasked
        time_sector = time_data[sector]
        time = np.asarray(time_sector, dtype=float)
        cadence_ref = cadence_bounds[sector][0]
        
        tjd_start = tjd_start_dict[sector]
        cadence_duration = 2 / (24 * 60)
        time_tjd = tjd_start + (time - time[0]) * cadence_duration

        # Manual binning
        bin_size = 0.004  # day decimal
        mask = ~np.isnan(time_tjd) & ~np.isnan(lc_sector) #for binning
        time_clean = time_tjd[mask] #without nans
        flux_clean = lc_sector[mask] #""     ""
        min_time = np.min(time_clean)
        max_time = np.max(time_clean)
        bins = np.arange(min_time, max_time + bin_size, bin_size)
        bin_indices = np.digitize(time_clean, bins)
        binned_time = []
        binned_flux = []
        for i in range(1, len(bins)):
            in_bin = bin_indices == i
            if np.any(in_bin):
                binned_time.append(np.mean(time_clean[in_bin]))
                binned_flux.append(np.mean(flux_clean[in_bin]))
        binned_time = np.array(binned_time)
        binned_flux = np.array(binned_flux)

        # optional BLS if not parameters
        
        lc = lk.LightCurve(time=binned_time, flux=binned_flux)
        lc_flat = lc.flatten(window_length=401).remove_outliers(sigma=30)
        period_grid = np.linspace(0.5, 20, 1000)
        lc_binned = lc_flat.bin(3.5)
        bls = lc_binned.to_periodogram(method='bls', period=period_grid, duration=0.4, frequency_factor=10)
        
        # 
        
        planet_period = float(bls.period_at_max_power.value)
        planet_t0 = float(bls.transit_time_at_max_power.value)
        planet_dur = float(bls.duration_at_max_power.value)

        p_val = even_odd_phase_our_data(binned_flux, period=planet_period, time=binned_time, t0=planet_t0, plot_size=0.1, same_axes=False, duration=planet_dur, plot=plot)
        sector_pvals.append(p_val)
        
    chi2_stat = -2 * np.sum(np.log(np.maximum(sector_pvals, 1e-10)))
    combined_p = 1 - chi2.cdf(chi2_stat, 2 * len(sector_pvals))
    even_odd_test_results.append((tid, combined_p))
        
print(even_odd_test_results)
