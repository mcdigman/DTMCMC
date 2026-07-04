"""Exact reference samplers for the test likelihoods (plan §4 Phase 2).

Needed by the NN-KL and CDF-error metrics. All samplers draw from an
explicit numpy Generator (DTMCMC.rng_helpers.get_rng) and never touch the
run RNG streams (plan D5). Each is exact by construction:

- gaussian: rejection of N(0, T·I) draws against the rectangular cutoff.
- cake: the two tiers are generalized-Gaussian radial profiles with
  analytic mixture weights (the amps), so r^e/(2 w^e) ~ Gamma(n/e)
  exactly; directions are isotropic; box truncation by rejection. This
  replaces the plan's numeric inverse-CDF route with an exact analytic
  transform of the same density (validated in tests against
  CakeLikelihood.get_loglike and analytic moments).
- eggbox: per-cell rejection with a proven envelope. The box tiles into
  cells around the points where every |cos| = 1; within an even-parity
  cell (product +1, the modes) the log-likelihood satisfies
  (p+1)^beta - 2^beta <= -a·u with u = r^2/2, via cos t <= exp(-t^2/2) on
  |t| <= pi/2 and a numerically-minimized slope a, giving a Gaussian
  envelope; odd-parity cells (p <= 0) use a constant envelope
  exp(1 - 2^beta). Cells are chosen proportional to their envelope
  integrals, so the sampler is exact over the whole box including the
  (utterly negligible, but not ignored) odd-cell mass.
"""

import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.special import gamma as gamma_func

from DTMCMC.likelihoods import eggbox as eggbox_module

if TYPE_CHECKING:
    from numpy.typing import NDArray

# cake tier constants matching DTMCMC/likelihoods/cake_likelihood.py, where
# they are currently function-local and cannot be imported; a test
# reconstructs the engine logL from these values so any drift fails CI.
# TODO(Phase 3): the tunable-cake work promotes the tier params to
# constructor args with identical defaults — import them from
# cake_likelihood then and delete these copies (single-sourcing).
CAKE_AMPS: tuple[float, float] = (0.5, 0.5)
CAKE_WIDTHS: tuple[float, float] = (4., 0.1)
CAKE_EXPONENTS: tuple[float, float] = (8., 2.)


def draw_truncated_gaussian(n_draws: int, n_par: int, cutoff: float, rng: np.random.Generator, T: float = 1.) -> NDArray[np.floating]:
    """Exact draws from N(0, T·I) truncated to the [-cutoff, cutoff]^n box."""
    scale = np.sqrt(T)
    out = np.zeros((n_draws, n_par))
    n_got = 0
    while n_got < n_draws:
        batch = rng.standard_normal((max(n_draws - n_got, 64), n_par)) * scale
        keep = batch[np.all(np.abs(batch) <= cutoff, axis=1)]
        n_take = min(keep.shape[0], n_draws - n_got)
        out[n_got:n_got + n_take] = keep[:n_take]
        n_got += n_take
    return out


def cake_moment_r2(n_par: int) -> float:
    """Analytic E[r^2] of the (untruncated) cake density.

    Per tier, r^e/(2 w^e) ~ Gamma(n/e) gives
    E[r^2] = w^2 · 2^(2/e) · Γ((n+2)/e) / Γ(n/e); mixture-weighted by the
    amps. Box truncation at the default cutoff is negligible (the tier-1
    tail beyond r=10 carries weight ~exp(-760)).
    """
    total = 0.
    for amp, width, exponent in zip(CAKE_AMPS, CAKE_WIDTHS, CAKE_EXPONENTS, strict=True):
        total += amp * width**2 * 2.**(2. / exponent) * gamma_func((n_par + 2.) / exponent) / gamma_func(n_par / exponent)
    return total


def draw_cake(n_draws: int, n_par: int, rng: np.random.Generator, cutoff: float = 10.) -> NDArray[np.floating]:
    """Exact draws from the cake posterior at T=1 in the [-cutoff, cutoff]^n box."""
    out = np.zeros((n_draws, n_par))
    n_got = 0
    while n_got < n_draws:
        n_want = max(n_draws - n_got, 64)
        tier = (rng.random(n_want) >= CAKE_AMPS[0]).astype(np.int64)
        widths = np.asarray(CAKE_WIDTHS)[tier]
        exponents = np.asarray(CAKE_EXPONENTS)[tier]
        s = rng.gamma(n_par / exponents)
        r = (2. * s)**(1. / exponents) * widths
        directions = rng.standard_normal((n_want, n_par))
        directions /= np.linalg.norm(directions, axis=1)[:, np.newaxis]
        batch = directions * r[:, np.newaxis]
        keep = batch[np.all(np.abs(batch) <= cutoff, axis=1)]
        n_take = min(keep.shape[0], n_draws - n_got)
        out[n_got:n_got + n_take] = keep[:n_take]
        n_got += n_take
    return out


@dataclass(frozen=True)
class EggboxCells:
    """Cell decomposition of the eggbox box: centers and parity."""

    centers: NDArray[np.floating]
    even_mask: NDArray[np.bool_]
    envelope_slope: float


def _eggbox_envelope_slope(n_par: int, betap: float) -> float:
    """Largest safe slope a with (1+e^-u)^beta - 2^beta <= -a·u on (0, u_max].

    u = r^2/2 ranges up to n·(pi/2)^2/2 within a cell; the ratio
    (2^beta - (1+e^-u)^beta)/u is minimized numerically on a dense grid
    with a multiplicative safety margin.
    """
    u_max = n_par * (np.pi / 2.)**2 / 2.
    us = np.linspace(1.e-6, u_max, 20001)
    slope = float(np.min((2.**betap - (1. + np.exp(-us))**betap) / us))
    return 0.999 * slope


def eggbox_cells(n_par: int) -> EggboxCells:
    """Enumerate all eggbox cells: centers where every |cos| = 1, with parity.

    Even-parity cells (product of cosines +1) hold the modes; the even
    count is cross-checked against eggbox.gen_nd_modelist in the tests.
    """
    per_dim = np.arange(-2, 3) * np.pi  # cos = +1 at even multiples, -1 at odd
    centers = np.array(list(itertools.product(per_dim, repeat=n_par)))
    signs = np.cos(centers).round().astype(np.int64)
    even_mask = signs.prod(axis=1) > 0
    return EggboxCells(centers=centers, even_mask=even_mask, envelope_slope=_eggbox_envelope_slope(n_par, eggbox_module.betap))


def eggbox_logL(samples: NDArray[np.floating]) -> NDArray[np.floating]:
    """Vectorized eggbox log-likelihood, tested against eggbox.get_loglike."""
    return (np.cos(samples).prod(axis=1) + 1.)**eggbox_module.betap


def draw_eggbox(n_draws: int, n_par: int, rng: np.random.Generator, max_batches: int = 100000) -> NDArray[np.floating]:
    """Exact draws from the eggbox posterior at T=1 over its full box."""
    cells = eggbox_cells(n_par)
    betap = eggbox_module.betap
    log_peak = 2.**betap
    slope = cells.envelope_slope

    n_even = int(np.count_nonzero(cells.even_mask))
    n_odd = cells.centers.shape[0] - n_even
    even_centers = cells.centers[cells.even_mask]
    odd_centers = cells.centers[~cells.even_mask]

    # envelope integrals decide the cell-type mixture: Gaussian envelope
    # over even cells, constant envelope exp(1 - 2^beta) over odd cells
    integral_even = n_even * (2. * np.pi / slope)**(n_par / 2.)
    integral_odd = n_odd * np.exp(1. - log_peak) * np.pi**n_par
    p_even = integral_even / (integral_even + integral_odd)

    out = np.zeros((n_draws, n_par))
    n_got = 0
    batch_size = 65536
    for _ in range(max_batches):
        if n_got >= n_draws:
            break
        pick_even = rng.random(batch_size) < p_even
        n_batch_even = int(np.count_nonzero(pick_even))

        # even cells: Gaussian proposal, rejected outside the cell, then
        # accepted with f/envelope = exp((p+1)^beta - 2^beta + slope*u)
        offsets = rng.standard_normal((n_batch_even, n_par)) / np.sqrt(slope)
        in_cell = np.all(np.abs(offsets) <= np.pi / 2., axis=1)
        u = (offsets**2).sum(axis=1) / 2.
        prod_cos = np.cos(offsets).prod(axis=1)
        log_accept = (prod_cos + 1.)**betap - log_peak + slope * u
        accept_even = in_cell & (np.log(rng.random(n_batch_even)) < log_accept)
        centers_even = even_centers[rng.integers(0, n_even, size=n_batch_even)]
        kept_even = centers_even[accept_even] + offsets[accept_even]

        # odd cells: uniform proposal, accept with exp((p+1)^beta - 1) <= 1
        n_batch_odd = batch_size - n_batch_even
        offsets_odd = rng.uniform(-np.pi / 2., np.pi / 2., size=(n_batch_odd, n_par))
        prod_cos_odd = np.cos(offsets_odd).prod(axis=1)
        # odd-parity cells flip the product's sign relative to offsets
        log_accept_odd = (-prod_cos_odd + 1.)**betap - 1.
        accept_odd = np.log(rng.random(n_batch_odd)) < log_accept_odd
        centers_odd = odd_centers[rng.integers(0, max(n_odd, 1), size=n_batch_odd)] if n_odd > 0 else np.zeros((n_batch_odd, n_par))
        kept_odd = (centers_odd[accept_odd] + offsets_odd[accept_odd]) if n_odd > 0 else np.zeros((0, n_par))

        kept = np.vstack([kept_even, kept_odd])
        n_take = min(kept.shape[0], n_draws - n_got)
        out[n_got:n_got + n_take] = kept[:n_take]
        n_got += n_take
    if n_got < n_draws:
        msg = f'eggbox rejection sampler produced only {n_got}/{n_draws} draws in {max_batches} batches'
        raise RuntimeError(msg)
    return out
