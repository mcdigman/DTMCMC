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

from DTMCMC.likelihoods import banana as banana_module
from DTMCMC.likelihoods import eggbox as eggbox_module
from DTMCMC.likelihoods import hyperpyramid as hyperpyramid_module
from DTMCMC.likelihoods.cake_likelihood import CAKE_DEFAULT_AMPS, CAKE_DEFAULT_EXPONENTS, CAKE_DEFAULT_WIDTHS

if TYPE_CHECKING:
    from numpy.typing import NDArray

# single-sourced from the engine (Phase 3 promoted the tier params to
# CakeLikelihood constructor args); the cross-check test still
# reconstructs the engine logL from these values to guard the density
# formula itself
CAKE_AMPS: tuple[float, ...] = tuple(float(amp) for amp in CAKE_DEFAULT_AMPS)
CAKE_WIDTHS: tuple[float, ...] = tuple(float(width) for width in CAKE_DEFAULT_WIDTHS)
CAKE_EXPONENTS: tuple[float, ...] = tuple(float(exponent) for exponent in CAKE_DEFAULT_EXPONENTS)


def draw_truncated_gaussian(
    n_draws: int, n_par: int, cutoff: float, rng: np.random.Generator, T: float = 1.0
) -> NDArray[np.floating]:
    """Exact draws from N(0, T·I) truncated to the [-cutoff, cutoff]^n box."""
    scale = np.sqrt(T)
    out = np.zeros((n_draws, n_par))
    n_got = 0
    while n_got < n_draws:
        batch = rng.standard_normal((max(n_draws - n_got, 64), n_par)) * scale
        keep = batch[np.all(np.abs(batch) <= cutoff, axis=1)]
        n_take = min(keep.shape[0], n_draws - n_got)
        out[n_got : n_got + n_take] = keep[:n_take]
        n_got += n_take
    return out


def cake_moment_r2(
    n_par: int,
    amps: tuple[float, ...] = CAKE_AMPS,
    widths: tuple[float, ...] = CAKE_WIDTHS,
    exponents: tuple[float, ...] = CAKE_EXPONENTS,
) -> float:
    """Analytic E[r^2] of the (untruncated) cake density.

    Per tier, r^e/(2 w^e) ~ Gamma(n/e) gives
    E[r^2] = w^2 · 2^(2/e) · Γ((n+2)/e) / Γ(n/e); mixture-weighted by the
    amps. Box truncation at the default cutoff is negligible (the tier-1
    tail beyond r=10 carries weight ~exp(-760)).
    """
    amps_arr = np.asarray(amps, dtype=np.float64)
    assert float(amps_arr.sum()) > 0.0
    # the engine accepts arbitrary amps; the posterior tier weights are
    # always amp_i / sum(amps), so normalize rather than assume sum = 1
    weights = amps_arr / amps_arr.sum()
    total = 0.0
    for weight, width, exponent in zip(weights, widths, exponents, strict=True):
        total += (
            weight
            * width**2
            * 2.0 ** (2.0 / exponent)
            * gamma_func((n_par + 2.0) / exponent)
            / gamma_func(n_par / exponent)
        )
    return float(total)


def draw_cake(
    n_draws: int,
    n_par: int,
    rng: np.random.Generator,
    cutoff: float = 10.0,
    amps: tuple[float, ...] = CAKE_AMPS,
    widths: tuple[float, ...] = CAKE_WIDTHS,
    exponents: tuple[float, ...] = CAKE_EXPONENTS,
) -> NDArray[np.floating]:
    """Exact draws from the cake posterior at T=1 in the [-cutoff, cutoff]^n box.

    Tier parameters default to the engine's cake; custom values sample
    the tunable cake family exactly. The engine accepts arbitrary amps
    (each tier integrates to exactly its amp, so the posterior tier
    weights are amp_i / sum(amps)); normalizing here keeps every
    engine-valid cake exactly sampleable.
    """
    amps_arr = np.asarray(amps, dtype=np.float64)
    widths_arr = np.asarray(widths, dtype=np.float64)
    exponents_arr = np.asarray(exponents, dtype=np.float64)
    assert float(amps_arr.sum()) > 0.0
    tier_cdf = np.cumsum(amps_arr / amps_arr.sum())

    out = np.zeros((n_draws, n_par))
    n_got = 0
    while n_got < n_draws:
        n_want = max(n_draws - n_got, 64)
        tier = np.searchsorted(tier_cdf, rng.random(n_want), side='right')
        tier = np.minimum(tier, amps_arr.size - 1)
        widths_pick = widths_arr[tier]
        exponents_pick = exponents_arr[tier]
        s = rng.gamma(n_par / exponents_pick)
        r = (2.0 * s) ** (1.0 / exponents_pick) * widths_pick
        directions = rng.standard_normal((n_want, n_par))
        directions /= np.linalg.norm(directions, axis=1)[:, np.newaxis]
        batch = directions * r[:, np.newaxis]
        keep = batch[np.all(np.abs(batch) <= cutoff, axis=1)]
        n_take = min(keep.shape[0], n_draws - n_got)
        out[n_got : n_got + n_take] = keep[:n_take]
        n_got += n_take
    return out


def cake_logL_radial(
    r: NDArray[np.floating],
    n_par: int,
    amps: tuple[float, ...] = CAKE_AMPS,
    widths: tuple[float, ...] = CAKE_WIDTHS,
    exponents: tuple[float, ...] = CAKE_EXPONENTS,
) -> NDArray[np.floating]:
    """Vectorized engine cake logL as a function of radius.

    The cake density is isotropic, so logL depends on r alone; this
    reconstruction from the tier constants is cross-validated against
    CakeLikelihood.get_loglike in the tests (the same guard pattern as
    test_cake_constants_match_engine).
    """
    r_arr = np.asarray(r, dtype=np.float64)
    dim_part = gamma_func(1.0 + n_par / 2.0) / np.pi ** (n_par / 2.0)
    out = np.full(r_arr.shape, -np.inf)
    for amp, width, exponent in zip(amps, widths, exponents, strict=True):
        tier_log = np.log(
            amp * dim_part / (2.0 ** (n_par / exponent) * width**n_par * gamma_func((exponent + n_par) / exponent))
        ) - r_arr**exponent / (2.0 * width**exponent)
        out = np.logaddexp(out, tier_log)
    return out


def cake_tempered_cumulants(
    betas: NDArray[np.floating],
    n_par: int,
    amps: tuple[float, ...] = CAKE_AMPS,
    widths: tuple[float, ...] = CAKE_WIDTHS,
    exponents: tuple[float, ...] = CAKE_EXPONENTS,
    r_max: float = 10.0,
    n_grid: int = 16384,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Exact tempered logL cumulants (E, Var) of an isotropic cake by radial quadrature.

    The tempered density at inverse temperature beta is
    exp(beta logL(r)) r^(n-1) dr up to normalization, so the first two
    logL cumulants — and hence heat capacity C = beta^2 Var and the
    entropy-ladder spacing integrals — follow from a 1D trapezoid over
    a dense radial grid, with no sampling anywhere. This is the
    likelihood-parametric profile used by the adaptive batteries.

    The quadrature integrates over the inscribed sphere r <= r_max
    rather than the prior box; the difference is the corner mass, which
    is negligible for temperatures with tier support well inside the
    box (widths * T^(1/exponent) << r_max) — the regime every ladder
    anchor in the tests uses. Validated against the measured gold arrays
    in the tests.
    """
    betas_arr = np.asarray(betas, dtype=np.float64)
    r_grid = np.linspace(1.0e-9, r_max, n_grid)
    logL_grid = cake_logL_radial(r_grid, n_par, amps, widths, exponents)
    log_shell = (n_par - 1.0) * np.log(r_grid)

    means = np.zeros(betas_arr.size)
    variances = np.zeros(betas_arr.size)
    for itrb, beta in enumerate(betas_arr):
        log_weight = beta * logL_grid + log_shell
        weight = np.exp(log_weight - log_weight.max())
        norm = np.trapezoid(weight, r_grid)
        mean = np.trapezoid(weight * logL_grid, r_grid) / norm
        second = np.trapezoid(weight * logL_grid**2, r_grid) / norm
        means[itrb] = mean
        variances[itrb] = max(second - mean**2, 0.0)
    return means, variances


def draw_banana(n_draws: int, n_par: int, rng: np.random.Generator) -> NDArray[np.floating]:
    """Exact draws from the banana posterior (constants from the likelihood module).

    The density factorizes exactly: v0 ~ N(0, 100), v1 | v0 ~
    N(100 B - B v0^2, 1), remaining coordinates ~ N(0, 1); the module's
    rectangular bounds are enforced by rejection (their truncation is
    negligible at the shipped constants but not assumed away).
    """
    if n_par < 2:
        msg = 'banana requires n_par >= 2'
        raise ValueError(msg)
    bananicity = float(banana_module.B)
    low_lims = np.full(n_par, banana_module.low_limn)
    high_lims = np.full(n_par, banana_module.high_limn)
    low_lims[:2] = banana_module.low_lim01
    high_lims[:2] = banana_module.high_lim01

    out = np.zeros((n_draws, n_par))
    n_got = 0
    while n_got < n_draws:
        n_want = max(n_draws - n_got, 64)
        batch = rng.standard_normal((n_want, n_par))
        batch[:, 0] *= 10.0
        batch[:, 1] += 100.0 * bananicity - bananicity * batch[:, 0] ** 2
        keep = batch[np.all((batch >= low_lims) & (batch <= high_lims), axis=1)]
        n_take = min(keep.shape[0], n_draws - n_got)
        out[n_got : n_got + n_take] = keep[:n_take]
        n_got += n_take
    return out


def hyperpyramid_marginal_variance() -> float:
    """Analytic per-coordinate variance of the 2D hyperpyramid posterior.

    With m = max|x_i|/sigma, the density depends only on m, so
    (m/sigma)^(1/s) ~ Gamma(n s) via the max-norm shell volume; at the
    module constants (n = 2, s = 1/2, sigma = 1) that is Exp(1), and a
    coordinate is on-face (variance m^2) or uniform (variance m^2/3)
    with equal probability: Var = E[m^2] (1/2 + 1/6) = 2/3.
    """
    n_par = 2
    s_exp = float(hyperpyramid_module.s)
    sigma = float(hyperpyramid_module.sigma)
    shape = n_par * s_exp
    # E[m^2] = sigma^2 E[t^(2s)] for t ~ Gamma(shape)
    moment_m2 = sigma**2 * gamma_func(shape + 2.0 * s_exp) / gamma_func(shape)
    return float(moment_m2 * (1.0 / 2.0 + 1.0 / 6.0))


def draw_hyperpyramid(n_draws: int, n_par: int, rng: np.random.Generator) -> NDArray[np.floating]:
    """Exact draws from the 2D hyperpyramid posterior.

    logL = -(max|x_i - center| / sigma)^(1/s): contours are max-norm
    spheres (squares), so draw the radial coordinate from the exact
    shell density — t = (m/sigma)^(1/s) ~ Gamma(n_par * s) — then place
    the point uniformly on the square shell (uniform face, uniform
    within-face coordinates). Box truncation handled by rejection (the
    mass beyond the module bounds is ~e^-225 at the shipped constants).
    """
    if n_par != 2:
        msg = 'hyperpyramid is 2D; n_par must be 2'
        raise ValueError(msg)
    s_exp = float(hyperpyramid_module.s)
    sigma = float(hyperpyramid_module.sigma)
    center = float(hyperpyramid_module.center)
    bound = float(hyperpyramid_module.high_lim)

    out = np.zeros((n_draws, n_par))
    n_got = 0
    while n_got < n_draws:
        n_want = max(n_draws - n_got, 64)
        m = sigma * rng.gamma(n_par * s_exp, size=n_want) ** s_exp
        face = rng.integers(0, 2 * n_par, size=n_want)
        batch = rng.uniform(-1.0, 1.0, size=(n_want, n_par)) * m[:, np.newaxis]
        batch[np.arange(n_want), face % n_par] = np.where(face < n_par, m, -m)
        batch += center
        keep = batch[np.all(np.abs(batch - center) <= bound, axis=1)]
        n_take = min(keep.shape[0], n_draws - n_got)
        out[n_got : n_got + n_take] = keep[:n_take]
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
    u_max = n_par * (np.pi / 2.0) ** 2 / 2.0
    us = np.linspace(1.0e-6, u_max, 20001)
    slope = float(np.min((2.0**betap - (1.0 + np.exp(-us)) ** betap) / us))
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
    return EggboxCells(
        centers=centers, even_mask=even_mask, envelope_slope=_eggbox_envelope_slope(n_par, eggbox_module.betap)
    )


def eggbox_logL(samples: NDArray[np.floating]) -> NDArray[np.floating]:
    """Vectorized eggbox log-likelihood, tested against eggbox.get_loglike."""
    return (np.cos(samples).prod(axis=1) + 1.0) ** eggbox_module.betap


def draw_eggbox(n_draws: int, n_par: int, rng: np.random.Generator, max_batches: int = 100000) -> NDArray[np.floating]:
    """Exact draws from the eggbox posterior at T=1 over its full box."""
    cells = eggbox_cells(n_par)
    betap = eggbox_module.betap
    log_peak = 2.0**betap
    slope = cells.envelope_slope

    n_even = int(np.count_nonzero(cells.even_mask))
    n_odd = cells.centers.shape[0] - n_even
    even_centers = cells.centers[cells.even_mask]
    odd_centers = cells.centers[~cells.even_mask]

    # envelope integrals decide the cell-type mixture: Gaussian envelope
    # over even cells, constant envelope exp(1 - 2^beta) over odd cells
    integral_even = n_even * (2.0 * np.pi / slope) ** (n_par / 2.0)
    integral_odd = n_odd * np.exp(1.0 - log_peak) * np.pi**n_par
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
        in_cell = np.all(np.abs(offsets) <= np.pi / 2.0, axis=1)
        u = (offsets**2).sum(axis=1) / 2.0
        prod_cos = np.cos(offsets).prod(axis=1)
        log_accept = (prod_cos + 1.0) ** betap - log_peak + slope * u
        accept_even = in_cell & (np.log(rng.random(n_batch_even)) < log_accept)
        centers_even = even_centers[rng.integers(0, n_even, size=n_batch_even)]
        kept_even = centers_even[accept_even] + offsets[accept_even]

        # odd cells: uniform proposal, accept with exp((p+1)^beta - 1) <= 1
        n_batch_odd = batch_size - n_batch_even
        offsets_odd = rng.uniform(-np.pi / 2.0, np.pi / 2.0, size=(n_batch_odd, n_par))
        prod_cos_odd = np.cos(offsets_odd).prod(axis=1)
        # odd-parity cells flip the product's sign relative to offsets
        log_accept_odd = (-prod_cos_odd + 1.0) ** betap - 1.0
        accept_odd = np.log(rng.random(n_batch_odd)) < log_accept_odd
        centers_odd = (
            odd_centers[rng.integers(0, max(n_odd, 1), size=n_batch_odd)]
            if n_odd > 0
            else np.zeros((n_batch_odd, n_par))
        )
        kept_odd = (centers_odd[accept_odd] + offsets_odd[accept_odd]) if n_odd > 0 else np.zeros((0, n_par))

        kept = np.vstack([kept_even, kept_odd])
        n_take = min(kept.shape[0], n_draws - n_got)
        out[n_got : n_got + n_take] = kept[:n_take]
        n_got += n_take
    if n_got < n_draws:
        msg = f'eggbox rejection sampler produced only {n_got}/{n_draws} draws in {max_batches} batches'
        raise RuntimeError(msg)
    return out
