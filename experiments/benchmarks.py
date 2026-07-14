"""Benchmark registry: reference draws and analytic anchors per likelihood.

One entry per harness likelihood name, so tests and analyses can ask a
single interface for "the ground truth about this target" instead of
special-casing likelihoods inline. An entry carries whatever ground
truth the target actually has — exact reference draws, analytic
per-coordinate moments, mode centers/weights — and omits what it does
not (for example, hawaii has none).

Reference-draw RNG conventions differ by construction and are flagged
per entry: the experiments-side samplers take an explicit numpy
Generator and never touch the run streams; the in-module gen_draws
samplers run under numba's global stream (uses_numba_stream=True), so
they are reproducible only relative to a seed_run point.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

from DTMCMC.likelihoods import ar1 as ar1_module
from DTMCMC.likelihoods import banana as banana_module
from DTMCMC.likelihoods import gaussian_mixture as gaussian_mixture_module
from DTMCMC.likelihoods import gaussian_shell as gaussian_shell_module
from DTMCMC.likelihoods import random_wheel as random_wheel_module
from DTMCMC.likelihoods import rosenbrock as rosenbrock_module
from DTMCMC.likelihoods import spoke_wheel as spoke_wheel_module
from experiments.reference_samplers import (
    cake_moment_r2,
    draw_banana,
    draw_cake,
    draw_eggbox,
    draw_hyperpyramid,
    draw_truncated_gaussian,
    hyperpyramid_marginal_variance,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


class DrawReference(Protocol):
    """Reference-posterior draw callable: (n_draws, n_par, rng) -> draws.

    Entries with uses_numba_stream=True ignore the Generator argument
    and consume numba's global stream instead (see module docstring).
    """

    def __call__(self, n_draws: int, n_par: int, rng: np.random.Generator) -> NDArray[np.floating]:
        """Draw (n_draws, n_par) exact posterior samples."""
        ...


@dataclass(frozen=True)
class BenchmarkTarget:
    """Ground truth available for one harness likelihood.

    Parameters
    ----------
    likelihood_name: str
        Key in the harness LIKELIHOOD_NAMES registry
    default_params: dict
        Canonical benchmark constructor params; reference draws and
        moments correspond to exactly these (a spec that overrides them
        must supply its own ground truth)
    draw_reference: DrawReference | None
        Exact posterior sampler, or None when the target has no ground
        truth
    uses_numba_stream: bool
        True for the in-module gen_draws samplers that consume numba's
        global stream rather than the passed Generator
    reference_moments: callable | None
        n_par -> (per-coordinate means, per-coordinate variances),
        exact for the default params unless the notes say otherwise
    mode_centers: NDArray | None
        (n_modes, n_par) mode centers for occupancy checks, where the
        target is a well-separated mixture
    mode_weights: NDArray | None
        Posterior weight of each mode, aligned with mode_centers
    notes: str
        Caveats a gate consumer must know (truncation approximations,
        missing ground truth, dimensionality constraints)
    """

    likelihood_name: str
    default_params: dict[str, float | int]
    draw_reference: DrawReference | None
    uses_numba_stream: bool = False
    reference_moments: Callable[[int], tuple[NDArray[np.floating], NDArray[np.floating]]] | None = None
    mode_centers: NDArray[np.floating] | None = None
    mode_weights: NDArray[np.floating] | None = None
    notes: str = ''


def _moments_constant(mean: float, var: float):
    """Per-coordinate moments callable for i.i.d.-coordinate targets."""

    def moments(n_par: int) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        return np.full(n_par, mean), np.full(n_par, var)

    return moments


def _moments_cake(n_par: int) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Cake per-coordinate moments: isotropic, so Var = E[r^2] / n_par."""
    return np.zeros(n_par), np.full(n_par, cake_moment_r2(n_par) / n_par)


def _moments_banana(n_par: int) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Banana moments: v0 ~ N(0, 100); v1 = 100B - B v0^2 + N(0,1) has mean 0
    and variance 1 + B^2 Var(v0^2) = 1 + B^2 (2 * 100^2); the rest are N(0, 1).
    """
    bananicity = float(banana_module.B)
    means = np.zeros(n_par)
    variances = np.ones(n_par)
    variances[0] = 100.0
    variances[1] = 1.0 + bananicity**2 * 2.0 * 100.0**2
    return means, variances


def _moments_rosenbrock(n_par: int) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Untruncated rosenbrock pair moments: x ~ N(1, 1/2), y | x ~ N(x^2, 1/200).

    E[y] = 3/2; Var[y] = Var(x^2) + 1/200 = (2 sigma^4 + 4 mu^2 sigma^2) + 1/200.
    The module's [-10, 10] box clips ~0.1% of the y = x^2 tail, so these
    are approximate at the 1e-3 level (the reference draws are exact).
    """
    means = np.tile([1.0, 1.5], n_par // 2)
    var_y = 2.0 * 0.5**2 + 4.0 * 1.0**2 * 0.5 + 1.0 / 200.0
    variances = np.tile([0.5, var_y], n_par // 2)
    return means, variances


def _moments_from_modes(centers: NDArray[np.floating], weights: NDArray[np.floating], width: float):
    """Exact per-coordinate moments of an equal-width isotropic Gaussian mixture."""

    def moments(n_par: int) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        assert n_par == centers.shape[1]
        means = weights @ centers
        second = weights @ (centers**2) + width**2
        return means, second - means**2

    return moments


def _numba_stream_draws(gen_draws):
    """Adapt an in-module gen_draws(n_draws, n_par) to the DrawReference signature."""

    def draw(n_draws: int, n_par: int, rng: np.random.Generator) -> NDArray[np.floating]:  # noqa: ARG001 — numba global stream, see module docstring
        return gen_draws(n_draws, n_par)

    return draw


_WHEEL_WEIGHTS = np.full(9, 1.0 / 9.0)
_SHELL_CENTERS = np.vstack([gaussian_shell_module.c1, gaussian_shell_module.c2])


def _moments_gaussian_mixture(n_par: int) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Exact moments of the 2-mode mixture: weight 1/3 at +5, 2/3 at -5, unit width."""
    mean = (1.0 / 3.0) * 5.0 + (2.0 / 3.0) * -5.0
    second = (1.0 / 3.0) * 25.0 + (2.0 / 3.0) * 25.0 + 1.0
    return np.full(n_par, mean), np.full(n_par, second - mean**2)


def mixture_mode_centers(n_par: int) -> NDArray[np.floating]:
    """The gaussian_mixture mode centers at the requested dimension."""
    return np.vstack([np.full(n_par, 5.0), np.full(n_par, -5.0)])


BENCHMARKS: dict[str, BenchmarkTarget] = {
    'gaussian': BenchmarkTarget(
        likelihood_name='gaussian',
        default_params={'n_par': 4, 'cutoff': 5},
        draw_reference=lambda n_draws, n_par, rng: draw_truncated_gaussian(n_draws, n_par, 5.0, rng),
        reference_moments=_moments_constant(0.0, 1.0),
        notes='moments quoted untruncated; cutoff=5 truncation is negligible',
    ),
    'cake': BenchmarkTarget(
        likelihood_name='cake',
        default_params={'n_par': 5, 'cutoff': 10},
        draw_reference=draw_cake,
        reference_moments=_moments_cake,
        notes='T=1 sits exactly at the tier phase transition; gold ladder data in data/*_cake_gold.npy',
    ),
    'eggbox': BenchmarkTarget(
        likelihood_name='eggbox',
        default_params={'n_par': 2},
        draw_reference=draw_eggbox,
        notes='mode accounting via reference_samplers.eggbox_cells (even-parity centers)',
    ),
    'hawaii': BenchmarkTarget(
        likelihood_name='hawaii',
        default_params={},
        draw_reference=None,
        notes='map-based 2D target with no reference sampler',
    ),
    'ar1': BenchmarkTarget(
        likelihood_name='ar1',
        default_params={'n_par': 50},
        draw_reference=_numba_stream_draws(ar1_module.gen_draws),
        uses_numba_stream=True,
        reference_moments=_moments_constant(0.0, 1.0),
        notes='stationary AR(1): unit marginal variance every coordinate; 10-sigma box truncation negligible',
    ),
    'banana': BenchmarkTarget(
        likelihood_name='banana',
        default_params={'n_par': 20},
        draw_reference=draw_banana,
        reference_moments=_moments_banana,
    ),
    'gaussian_mixture': BenchmarkTarget(
        likelihood_name='gaussian_mixture',
        default_params={'n_par': 50},
        draw_reference=_numba_stream_draws(gaussian_mixture_module.gen_draws),
        uses_numba_stream=True,
        reference_moments=_moments_gaussian_mixture,
        mode_centers=mixture_mode_centers(50),
        mode_weights=np.array([1.0 / 3.0, 2.0 / 3.0]),
        notes='mode centers quoted at the default n_par; rebuild with mixture_mode_centers for other dims',
    ),
    'gaussian_shell': BenchmarkTarget(
        likelihood_name='gaussian_shell',
        default_params={'n_par': 2},
        draw_reference=_numba_stream_draws(gaussian_shell_module.gen_draws),
        uses_numba_stream=True,
        mode_centers=_SHELL_CENTERS,
        mode_weights=np.array([0.5, 0.5]),
        notes='two rings, not point modes: occupancy by nearest shell center',
    ),
    'hyperpyramid': BenchmarkTarget(
        likelihood_name='hyperpyramid',
        default_params={'n_par': 2},
        draw_reference=draw_hyperpyramid,
        reference_moments=_moments_constant(0.0, hyperpyramid_marginal_variance()),
    ),
    'random_wheel': BenchmarkTarget(
        likelihood_name='random_wheel',
        default_params={'n_par': 2},
        draw_reference=_numba_stream_draws(random_wheel_module.gen_draws),
        uses_numba_stream=True,
        reference_moments=_moments_from_modes(
            np.asarray(random_wheel_module.cs, dtype=np.float64), _WHEEL_WEIGHTS, float(random_wheel_module.w)
        ),
        mode_centers=np.asarray(random_wheel_module.cs, dtype=np.float64),
        mode_weights=_WHEEL_WEIGHTS,
    ),
    'rosenbrock': BenchmarkTarget(
        likelihood_name='rosenbrock',
        default_params={'n_par': 20},
        draw_reference=_numba_stream_draws(rosenbrock_module.gen_draws),
        uses_numba_stream=True,
        reference_moments=_moments_rosenbrock,
        notes='moments untruncated (see _moments_rosenbrock); reference draws are exact under the box',
    ),
    'spoke_wheel': BenchmarkTarget(
        likelihood_name='spoke_wheel',
        default_params={'n_par': 2},
        draw_reference=_numba_stream_draws(spoke_wheel_module.gen_draws),
        uses_numba_stream=True,
        reference_moments=_moments_from_modes(
            np.asarray(spoke_wheel_module.cs, dtype=np.float64), _WHEEL_WEIGHTS, float(spoke_wheel_module.w)
        ),
        mode_centers=np.asarray(spoke_wheel_module.cs, dtype=np.float64),
        mode_weights=_WHEEL_WEIGHTS,
    ),
}


def mode_occupancy(samples: NDArray[np.floating], centers: NDArray[np.floating]) -> NDArray[np.floating]:
    """Fraction of samples nearest each mode center (Euclidean assignment)."""
    distances = np.linalg.norm(samples[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
    nearest = np.argmin(distances, axis=1)
    return np.bincount(nearest, minlength=centers.shape[0]) / samples.shape[0]
