import math

import pytest

np = pytest.importorskip("numpy")

from multi_sector_gather import (
    MultiSectorGrid,
    SectorTransitStats,
    _map_epoch_index,
    combine_sector_statistics,
    expand_with_cadence_gaps,
    gather_sector_statistics_cpu,
)
from util import box_transit


def test_box_transit_detection_peak_aligned_to_injected_signal():
    rng = np.random.default_rng(12345)

    delta_time = 1.0
    half_N = 18
    cadence_times = np.arange(2 * half_N, dtype=float) * delta_time

    period_steps = 7
    duration_steps = 2
    expected_period = period_steps * delta_time
    t0 = 3.0

    noise = rng.normal(0.0, 0.001, size=cadence_times.size)
    injected = box_transit(
        cadence_times,
        period=expected_period,
        dur=duration_steps * delta_time,
        t0=t0,
        alpha=0.02,
    )
    lightcurve = noise + injected

    y_d = lightcurve.astype(float)
    K_d = np.ones((cadence_times.size, cadence_times.size), dtype=float)

    numerator, denominator = gather_sector_statistics_cpu(y_d, K_d)
    sector_stats = SectorTransitStats(start_time=0.0, numerator=numerator, denominator=denominator)

    grid = MultiSectorGrid(
        period_grid=np.arange(1, numerator.shape[0] + 1, dtype=float) * delta_time,
        delta_time=delta_time,
        epoch_reference=0.0,
    )

    _, _, detection = combine_sector_statistics(grid, [sector_stats])

    best_period_idx, best_epoch_idx = np.unravel_index(np.argmax(detection), detection.shape)

    assert math.isclose(grid.period_grid[best_period_idx], expected_period)

    expected_epoch = int(round((t0 % expected_period) / delta_time)) % period_steps
    assert best_epoch_idx == expected_epoch


def test_expand_with_cadence_gaps_zero_fills_missing_samples():
    sector = 99
    cadence_bounds = {sector: (100, 110)}
    time_data = np.array([100, 102, 105, 109])
    y_d = np.array([1.0, -1.0, 0.5, 2.0])
    K_d = np.eye(len(y_d)) * 3.0

    padded_y, padded_K = expand_with_cadence_gaps(y_d, K_d, time_data, sector=sector, cadence_bounds=cadence_bounds)

    assert padded_y.shape[0] == 10
    assert np.allclose(padded_y[[0, 2, 5, 9]], y_d)
    missing_mask = np.ones_like(padded_y, dtype=bool)
    missing_mask[[0, 2, 5, 9]] = False
    assert np.count_nonzero(padded_y[missing_mask]) == 0

    expected_slice = padded_K[np.ix_([0, 2, 5, 9], [0, 2, 5, 9])]
    assert np.allclose(expected_slice, K_d)
    off_diagonal_mask = np.ones_like(padded_K, dtype=bool)
    off_diagonal_mask[np.ix_([0, 2, 5, 9], [0, 2, 5, 9])] = False
    assert np.count_nonzero(padded_K[off_diagonal_mask]) == 0


def test_map_epoch_index_wraps_on_period_boundaries():
    period = 4.0
    delta_time = 1.0
    epoch_reference = 0.0
    sector_start_time = 0.0

    assert _map_epoch_index(period, delta_time, epoch_reference, sector_start_time, global_epoch_idx=0) == 0
    # Exactly one full period later should wrap to epoch 0 without drifting.
    assert _map_epoch_index(period, delta_time, epoch_reference, sector_start_time, global_epoch_idx=4) == 0
    # Half-period offset should land in the middle of the epoch grid.
    assert _map_epoch_index(period, delta_time, epoch_reference, sector_start_time, global_epoch_idx=2) == 2
