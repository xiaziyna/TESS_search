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
from typing import Dict, List, Tuple

import numpy as np
import pycuda.autoinit  # noqa: F401  # ensures CUDA context is created
import pycuda.driver as drv
from pycuda.compiler import SourceModule


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


def _compile_gather_kernels() -> Tuple[drv.Function, drv.Function]:
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


def _map_epoch_index(
    period: float, delta_time: float, epoch_reference: float, sector_start_time: float, global_epoch_idx: int
) -> int:
    """Map a global epoch index onto the local sector epoch index."""

    epoch_count = int(round(period / delta_time))
    global_epoch_time = epoch_reference + (global_epoch_idx * delta_time)
    phase = math.fmod(global_epoch_time - sector_start_time + period, period)
    return int(round(phase / delta_time)) % epoch_count


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


__all__ = [
    "SectorTransitStats",
    "MultiSectorGrid",
    "bundle_sector_stats",
    "combine_sector_statistics",
    "gather_sector_statistics",
    "example_usage",
]
