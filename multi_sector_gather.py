"""
Prototype utilities for combining GPU gather transit statistics across
multiple TESS sectors.

This module follows the gather approach demonstrated in ``pycuda_exosearch.ipynb``
and adds a light-weight combiner that aligns period/epoch statistics from
individual sectors onto a common global grid using their observation times.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Tuple

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - only for type hints without GPU import
    import pycuda.driver as drv


GATHER_KERNEL = r"""
__global__ void gather_transit_num(float *output, const float *input, int half_N)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x; // epoch index
    int ty = blockIdx.y * blockDim.y + threadIdx.y; // period index offset by +1
    int p = ty + 1;
    int no_transit = floorf(((2 * half_N) - tx - 1) / p) + 1;
    float value = 0.0f;

    if (tx < p && p <= half_N)
    {
        for (int i = 0; i < no_transit; ++i)
        {
            value += input[(i * p) + tx];
        }
        output[(half_N * ty) + tx] = value;
    }
}

__global__ void gather_transit_den(float *output, const float *input, int half_N)
{ // Redundant (simple) version
    int tx = blockIdx.x * blockDim.x + threadIdx.x; // epoch index
    int ty = blockIdx.y * blockDim.y + threadIdx.y; // period index offset by +1
    int p = ty + 1;
    int t0 = tx;
    int no_transit = floorf(((2 * half_N) - t0 - 1) / p) + 1;
    float value = 0.0f;

    if (t0 < p && p <= half_N)
    {
        for (int i = 0; i < no_transit; ++i)
        {
            for (int j = 0; j < no_transit; ++j)
            {
                value += input[((i * p) + tx) * (2 * half_N) + (j * p) + tx];
            }
        }
        output[(half_N * (p - 1)) + t0] = value;
    }
}
"""


def _compile_gather_kernels() -> Tuple[object, object]:
    from pycuda.compiler import SourceModule

    mod = SourceModule(GATHER_KERNEL)
    return mod.get_function("gather_transit_num"), mod.get_function("gather_transit_den")


@dataclass
class SectorTransitStats:
    """Container for per-sector gather statistics."""

    start_time: float
    """BJD of the first cadence in the sector."""

    numerator: np.ndarray
    """Gathered ``y^T K t`` values with shape (num_period, num_period)."""

    denominator: np.ndarray
    """Gathered ``t^T K t`` values with shape (num_period, num_period)."""


@dataclass
class MultiSectorGrid:
    """Shared grid definition used for combining statistics."""

    period_grid: np.ndarray
    """Array of trial periods in days (or the same units as ``delta_time``)."""

    delta_time: float
    """Step size between cadences (days). Should match the ``delta`` used on GPU."""

    epoch_reference: float
    """Absolute reference time (days) used to define epoch index 0."""

    @property
    def max_epoch_count(self) -> int:
        return int(np.ceil(np.max(self.period_grid) / self.delta_time))


def gather_sector_statistics(y_d: np.ndarray, K_d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Run gather kernels for a single sector.

    Parameters
    ----------
    y_d
        1D array of ``y^T Cov_inv`` convolved with the transit profile and
        sampled at the cadence step ``delta``. Length should be ``2 * half_N``.
    K_d
        2D array containing the convolved covariance ``Cov_inv * (t t^T)``
        on the same grid as ``y_d``. Shape should be ``(2 * half_N, 2 * half_N)``.

    Returns
    -------
    numerator, denominator : tuple of ``np.ndarray``
        Each has shape ``(half_N, half_N)``; only the first ``p`` entries along
        the epoch axis are populated for a given period index ``p``.
    """

    if y_d.ndim != 1:
        raise ValueError("y_d must be a 1D array")
    if K_d.shape[0] != K_d.shape[1]:
        raise ValueError("K_d must be square")
    if y_d.shape[0] != K_d.shape[0]:
        raise ValueError("y_d and K_d must represent the same cadence grid")
    if y_d.shape[0] % 2:
        raise ValueError("gather kernels expect an even-length cadence grid")

    import pycuda.autoinit  # noqa: F401  # ensures CUDA context is created
    import pycuda.driver as drv

    half_N = y_d.shape[0] // 2
    gather_num, gather_den = _compile_gather_kernels()

    output_num = np.zeros((half_N, half_N), dtype=np.float32)
    output_den = np.zeros((half_N, half_N), dtype=np.float32)

    block_dim = (32, 8, 1)
    grid_dim = (
        int(math.ceil(half_N / block_dim[0])),
        int(math.ceil(half_N / block_dim[1])),
    )

    gather_num(
        drv.Out(output_num),
        drv.In(y_d.astype(np.float32)),
        np.int32(half_N),
        block=block_dim,
        grid=grid_dim,
    )
    gather_den(
        drv.Out(output_den),
        drv.In(K_d.astype(np.float32)),
        np.int32(half_N),
        block=block_dim,
        grid=grid_dim,
    )
    return output_num, output_den


def gather_sector_statistics_cpu(y_d: np.ndarray, K_d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """CPU implementation mirroring :func:`gather_sector_statistics` for testing.

    This follows the same indexing logic as the CUDA kernels to make it possible
    to validate end-to-end period/epoch searches without requiring a GPU.
    """

    if y_d.ndim != 1:
        raise ValueError("y_d must be a 1D array")
    if K_d.shape[0] != K_d.shape[1]:
        raise ValueError("K_d must be square")
    if y_d.shape[0] != K_d.shape[0]:
        raise ValueError("y_d and K_d must represent the same cadence grid")
    if y_d.shape[0] % 2:
        raise ValueError("gather kernels expect an even-length cadence grid")

    series_len = y_d.shape[0]
    half_N = series_len // 2
    output_num = np.zeros((half_N, half_N), dtype=np.float64)
    output_den = np.zeros((half_N, half_N), dtype=np.float64)

    for ty in range(half_N):
        p = ty + 1
        max_epoch = min(p, half_N)
        for tx in range(max_epoch):
            no_transit = ((series_len - tx - 1) // p) + 1
            num_val = 0.0
            den_val = 0.0
            for i in range(no_transit):
                idx_i = (i * p) + tx
                num_val += y_d[idx_i]
                for j in range(no_transit):
                    idx_j = (j * p) + tx
                    den_val += K_d[idx_i, idx_j]
            output_num[ty, tx] = num_val
            output_den[ty, tx] = den_val

    return output_num, output_den


def _map_epoch_index(
    period: float, delta_time: float, epoch_reference: float, sector_start_time: float, global_epoch_idx: int
) -> int:
    """Map a global epoch index onto the local sector epoch index."""

    epoch_count = max(1, int(round(period / delta_time)))
    global_epoch_time = epoch_reference + (global_epoch_idx * delta_time)
    phase = (global_epoch_time - sector_start_time) % period
    local_epoch = int(round(phase / delta_time)) % epoch_count
    return local_epoch


def combine_sector_statistics(
    grid: MultiSectorGrid, sector_results: List[SectorTransitStats]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Combine gather statistics from multiple sectors onto a shared grid.

    The per-period/epoch numerator and denominator values are phase-aligned
    using the absolute start time of each sector. Epoch index 0 corresponds to
    ``grid.epoch_reference`` and increases in steps of ``grid.delta_time``.

    Parameters
    ----------
    grid
        Shared grid definition containing the period grid, epoch reference, and
        cadence spacing used in the GPU gather pass.
    sector_results
        List of per-sector results produced by :func:`gather_sector_statistics`.

    Returns
    -------
    numerator, denominator, detection : tuple of ``np.ndarray``
        Aggregated numerator and denominator along with the combined detection
        statistic ``num / sqrt(den)``. Arrays have shape
        ``(len(period_grid), grid.max_epoch_count)``; entries beyond the period-
        dependent epoch count are zero.
    """

    num_global = np.zeros((len(grid.period_grid), grid.max_epoch_count), dtype=np.float64)
    den_global = np.zeros_like(num_global)

    for sector in sector_results:
        for p_idx, period in enumerate(grid.period_grid):
            epoch_count = int(round(period / grid.delta_time))
            for global_epoch_idx in range(epoch_count):
                local_idx = _map_epoch_index(
                    period=period,
                    delta_time=grid.delta_time,
                    epoch_reference=grid.epoch_reference,
                    sector_start_time=sector.start_time,
                    global_epoch_idx=global_epoch_idx,
                )
                num_global[p_idx, global_epoch_idx] += sector.numerator[p_idx, local_idx]
                den_global[p_idx, global_epoch_idx] += sector.denominator[p_idx, local_idx]

    detection = np.divide(
        num_global,
        np.sqrt(den_global),
        out=np.zeros_like(num_global),
        where=den_global != 0.0,
    )
    return num_global, den_global, detection


def bundle_sector_stats(
    start_time: float, y_d: np.ndarray, K_d: np.ndarray
) -> SectorTransitStats:
    """Convenience helper that runs the gather kernels and returns a dataclass."""

    numerator, denominator = gather_sector_statistics(y_d=y_d, K_d=K_d)
    return SectorTransitStats(start_time=start_time, numerator=numerator, denominator=denominator)


def expand_with_cadence_gaps(
    y_d: np.ndarray, K_d: np.ndarray, time_data: np.ndarray, sector: int, cadence_bounds: Dict[int, Tuple[int, int]]
) -> Tuple[np.ndarray, np.ndarray]:
    """Pad sparse gather inputs with zeros to the full cadence range for a sector.

    Parameters
    ----------
    y_d, K_d
        Convolved gather inputs defined only at observed cadences (gaps removed).
    time_data
        Cadence numbers corresponding to the entries in ``y_d``/``K_d``.
    sector
        Sector identifier used to look up cadence start/end in ``cadence_bounds``.
    cadence_bounds
        Mapping from sector to ``(cadence_start, cadence_end)``.

    Returns
    -------
    padded_y, padded_K : tuple of ``np.ndarray``
        Zero-padded arrays whose shapes match the full cadence span for the sector,
        suitable for passing to :func:`gather_sector_statistics`.
    """

    if sector not in cadence_bounds:
        raise KeyError(f"Cadence bounds missing for sector {sector}")

    cadence_start, cadence_end = cadence_bounds[sector]
    cadence_span = cadence_end - cadence_start
    if cadence_span <= 0:
        raise ValueError(f"Invalid cadence bounds for sector {sector}: {cadence_bounds[sector]}")

    offsets = np.asarray(time_data) - cadence_start
    if offsets.shape[0] != y_d.shape[0]:
        raise ValueError("time_data and y_d must describe the same number of cadences")
    if K_d.shape != (y_d.shape[0], y_d.shape[0]):
        raise ValueError("K_d must be square and aligned with y_d")
    if np.any(offsets < 0) or np.any(offsets >= cadence_span):
        raise ValueError("time_data contains cadences outside the expected sector bounds")

    padded_y = np.zeros(cadence_span, dtype=y_d.dtype)
    padded_K = np.zeros((cadence_span, cadence_span), dtype=K_d.dtype)

    padded_y[offsets] = y_d
    idx = np.ix_(offsets, offsets)
    padded_K[idx] = K_d

    return padded_y, padded_K


def example_usage(sector_pickles: Dict[int, dict], delta_cadence_minutes: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Example wiring for combining sector statistics.

    Parameters
    ----------
    sector_pickles
        Dictionary keyed by sector number containing the loaded pickle content
        produced by the TESS preprocessing pipeline. Each entry must include
        ``"time_data"`` (in days), and the pre-computed gather inputs
        ``"y_d"`` and ``"K_d"`` for the target transit duration.
    delta_cadence_minutes
        Cadence spacing used when computing ``y_d`` and ``K_d``; defaults to
        the 2-minute TESS cadence.

    Returns
    -------
    Combined numerator, denominator, and detection arrays.
    """

    cad_per_day = 24.0 * 60.0 / delta_cadence_minutes
    delta_time = 1.0 / cad_per_day

    period_grid = np.arange(1, sector_pickles[sorted(sector_pickles.keys())[0]]["y_d"].shape[0] // 2 + 1) * delta_time
    epoch_reference = min(sector_pickles[s]["time_data"][0] for s in sector_pickles)
    grid = MultiSectorGrid(period_grid=period_grid, delta_time=delta_time, epoch_reference=epoch_reference)

    sector_results = []
    for sector_id, payload in sector_pickles.items():
        sector_results.append(bundle_sector_stats(start_time=payload["time_data"][0], y_d=payload["y_d"], K_d=payload["K_d"]))

    return combine_sector_statistics(grid, sector_results)


def load_augmented_sector_from_pickle(
    tic_id: str, sector: int | None = None, data_dir: str = "~/TESS/data"
) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Load a pickle light curve and zero-fill gather inputs for one sector.

    This helper demonstrates how to expand convolved gather inputs that were
    computed only on observed cadences. The current implementation reuses the
    stored ``detrend_data`` array for ``y_d`` and builds a diagonal ``K_d``
    placeholder with the same cadence coverage purely for illustrative
    purposes; replace these arrays with the convolved versions your pipeline
    produces before calling :func:`gather_sector_statistics`.

    Parameters
    ----------
    tic_id
        TIC identifier corresponding to ``<data_dir>/<tic_id>.p``.
    sector
        Optional sector to extract. If omitted the earliest available sector is
        used.
    data_dir
        Directory containing the pickle files created by
        ``download_lc_julias_version``.

    Returns
    -------
    sector_used, padded_y, padded_K, cadence_numbers
        The sector identifier selected along with zero-padded ``y_d``/``K_d``
        arrays and the cadence numbers for that sector.
    """

    import os
    import pickle

    from info import cadence_bounds

    payload_path = os.path.expanduser(f"{data_dir.rstrip('/')}/{tic_id}.p")
    with open(payload_path, "rb") as handle:
        payload = pickle.load(handle)

    (
        lc_data,
        processed_lc_data,
        detrend_data,
        norm_offset,
        quality_data,
        time_data,
        cam_data,
        ccd_data,
        coeff_ls,
        centroid_xy_data,
        *rest,
    ) = payload

    available_sectors = sorted(time_data.keys())
    if not available_sectors:
        raise ValueError("No sector data present in the pickle payload")

    sector_used = sector if sector is not None else available_sectors[0]
    if sector_used not in time_data:
        raise KeyError(f"Sector {sector_used} not found in the pickle payload")

    y_sparse = detrend_data[sector_used]
    cadence_numbers = time_data[sector_used]
    # Placeholder covariance aligned with the detrended flux for demonstration.
    K_sparse = np.eye(len(y_sparse), dtype=float)

    padded_y, padded_K = expand_with_cadence_gaps(
        y_d=y_sparse, K_d=K_sparse, time_data=cadence_numbers, sector=sector_used, cadence_bounds=cadence_bounds
    )

    return sector_used, padded_y, padded_K, cadence_numbers


__all__ = [
    "SectorTransitStats",
    "MultiSectorGrid",
    "expand_with_cadence_gaps",
    "bundle_sector_stats",
    "gather_sector_statistics_cpu",
    "combine_sector_statistics",
    "gather_sector_statistics",
    "example_usage",
    "load_augmented_sector_from_pickle",
]
