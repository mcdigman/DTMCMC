"""Composable posterior-recovery and ladder-structure gates.

These gates compare run samples and ladder structure against supplied
reference data: exact reference draws, analytic moments, mode weights,
and quadrature entropy profiles. Nothing here is likelihood-specific;
thresholds and reference data are provided by the caller.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from experiments.benchmarks import mode_occupancy
from experiments.metrics import nn_divergence_symmetric

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class GateReport:
    """Accumulated gate outcomes: prose violations plus the measured stats."""

    violations: list[str] = field(default_factory=list)
    stats: dict[str, float] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """Whether every gate held."""
        return not self.violations

    def merge(self, other: GateReport) -> GateReport:
        """Fold another report's violations and stats into this one."""
        self.violations.extend(other.violations)
        self.stats.update(other.stats)
        return self


def dedup_rows(samples: NDArray[np.floating]) -> NDArray[np.floating]:
    """Unique sample rows: repeated (rejected-move) states break the NN estimator.

    Zero nearest-neighbor distances drive the NN entropy estimator to
    infinity, so chain output must be deduplicated before any NN gate.
    """
    return np.unique(samples, axis=0)


def moment_gates(
        samples: NDArray[np.floating],
        means_ref: NDArray[np.floating],
        vars_ref: NDArray[np.floating],
        *,
        mean_tol_sigmas: float,
        var_ratio_bounds: tuple[float, float],
        label: str = 'moments',
) -> GateReport:
    """Per-coordinate first two cumulants against reference values.

    mean_tol_sigmas is in units of the reference standard deviation, so
    one tolerance serves coordinates of very different scales.
    """
    report = GateReport()
    scale = np.sqrt(vars_ref)
    mean_dev = np.abs(samples.mean(axis=0) - means_ref) / scale
    var_ratio = samples.var(axis=0) / vars_ref
    report.stats[f'{label}_mean_dev_max'] = float(mean_dev.max())
    report.stats[f'{label}_var_ratio_min'] = float(var_ratio.min())
    report.stats[f'{label}_var_ratio_max'] = float(var_ratio.max())
    if float(mean_dev.max()) > mean_tol_sigmas:
        report.violations.append(f'{label}: max per-coordinate mean deviation {mean_dev.max():.3f} sigma > {mean_tol_sigmas}')
    if float(var_ratio.min()) < var_ratio_bounds[0] or float(var_ratio.max()) > var_ratio_bounds[1]:
        report.violations.append(f'{label}: per-coordinate variance ratios [{var_ratio.min():.3f}, {var_ratio.max():.3f}] outside {var_ratio_bounds}')
    return report


def nn_gate(
        reference: NDArray[np.floating],
        samples: NDArray[np.floating],
        *,
        threshold: float,
        n_use: int,
        rng: np.random.Generator,
        label: str = 'nn',
) -> GateReport:
    """Symmetric NN divergence below a calibrated threshold.

    Symmetric by construction (max of both orientations): the signed
    statistic can be negative for overconcentrated samples. Samples
    must already be deduplicated.
    """
    report = GateReport()
    value = float(nn_divergence_symmetric(reference, samples, n_use, rng))
    report.stats[f'{label}_sym'] = value
    if not np.isfinite(value) or value > threshold:
        report.violations.append(f'{label}: symmetric NN divergence {value:.3f} > {threshold}')
    return report


def occupancy_gates(
        samples: NDArray[np.floating],
        centers: NDArray[np.floating],
        weights_ref: NDArray[np.floating],
        *,
        tol: float,
        label: str = 'occupancy',
) -> GateReport:
    """Nearest-center mode occupancy within tol of the reference weights."""
    report = GateReport()
    occupancy = mode_occupancy(samples, centers)
    deviation = np.abs(occupancy - weights_ref)
    report.stats[f'{label}_dev_max'] = float(deviation.max())
    if float(deviation.max()) > tol:
        report.violations.append(f'{label}: mode occupancy deviates up to {deviation.max():.3f} > {tol} (occupancy {np.round(occupancy, 3).tolist()})')
    return report


def radial_mixture_gates(
        samples: NDArray[np.floating],
        *,
        r2_threshold: float,
        narrow_frac_ref: float,
        narrow_frac_tol: float,
        r2_mean_ref: float,
        r2_mean_rtol: float,
        min_tier_flips: int,
        label: str = 'tiers',
) -> GateReport:
    """Concentric radial-mixture (cake-style) tier recovery.

    Checks the narrow-tier fraction, the radial second moment, and that
    the stored series actually interconverts between tiers (a sampler
    stuck in one tier can fake a good fraction only by luck, never the
    flip count).
    """
    report = GateReport()
    r2 = (samples**2).sum(axis=1)
    narrow_mask = r2 < r2_threshold
    narrow_frac = float(narrow_mask.mean())
    n_flips = int(np.count_nonzero(np.diff(narrow_mask.astype(np.int8))))
    r2_mean = float(r2.mean())
    report.stats[f'{label}_narrow_frac'] = narrow_frac
    report.stats[f'{label}_n_flips'] = float(n_flips)
    report.stats[f'{label}_r2_mean'] = r2_mean
    if abs(narrow_frac - narrow_frac_ref) > narrow_frac_tol:
        report.violations.append(f'{label}: narrow fraction {narrow_frac:.3f} outside {narrow_frac_ref} +- {narrow_frac_tol}')
    if abs(r2_mean - r2_mean_ref) > r2_mean_rtol * r2_mean_ref:
        report.violations.append(f'{label}: E[r^2] {r2_mean:.3f} outside {r2_mean_ref:.3f} +- {r2_mean_rtol:.0%}')
    if n_flips < min_tier_flips:
        report.violations.append(f'{label}: only {n_flips} tier flips < {min_tier_flips} (tiers not interconverting)')
    return report


def ladder_entropy_gates(
        finite_Ts: NDArray[np.floating],
        betas_profile: NDArray[np.floating],
        s_profile: NDArray[np.floating],
        *,
        tip_max_nats: float,
        link_max_nats: float,
        label: str = 'ladder',
) -> GateReport:
    """Ladder structure against a reference entropy profile S(beta).

    The profile (descending beta, cumulative nats) comes from analytic
    quadrature or a trusted measured run; rungs outside its temperature
    span are not judged (extrapolating the profile would invent
    entropy). Gates: the coldest link may carry at most tip_max_nats,
    and every other in-span link at most link_max_nats.
    """
    report = GateReport()
    span = (finite_Ts >= 1. / betas_profile.max()) & (finite_Ts <= 1. / betas_profile.min())
    Ts_in_span = np.sort(finite_Ts[span])
    betas_rungs = 1. / Ts_in_span
    s_at_rungs = np.asarray(np.interp(betas_rungs[::-1], betas_profile[::-1], s_profile[::-1]))[::-1]
    increments = np.abs(np.diff(s_at_rungs))
    if increments.size == 0:
        report.violations.append(f'{label}: fewer than two rungs inside the reference profile span')
        return report
    report.stats[f'{label}_tip_nats'] = float(increments[0])
    report.stats[f'{label}_max_link_nats'] = float(increments[1:].max()) if increments.size > 1 else 0.
    if float(increments[0]) > tip_max_nats:
        report.violations.append(f'{label}: coldest link hides {increments[0]:.2f} nats > {tip_max_nats}')
    if increments.size > 1 and float(increments[1:].max()) > link_max_nats:
        report.violations.append(f'{label}: a non-tip link hides {increments[1:].max():.2f} nats > {link_max_nats}')
    return report
