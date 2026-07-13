import numpy as np
from numpy.testing import assert_allclose

from DTMCMC.chain_analysis_helpers import StoreView
from DTMCMC.corr_summary_helpers import CorrelationSummary, autocorr_helper, get_crosscorr_sum


def make_store_view(samples_store: np.ndarray, logLs_store: np.ndarray, Ts: np.ndarray, block_size: int = 16) -> StoreView:
    n_rows, n_cold, n_par = samples_store.shape
    return StoreView(
        samples_store=samples_store,
        logLs_store=logLs_store,
        Ts=Ts,
        store_size=n_rows,
        n_cold=n_cold,
        n_chain=n_cold,
        block_size=block_size,
        store_thin=1,
        n_par=n_par,
        itrn=n_rows,
    )


def test_crosscorr_sum_uses_stored_sample_count() -> None:
    rng = np.random.default_rng(0)
    samples_store = rng.standard_normal((128, 2, 1))
    logLs_store = np.zeros((128, 2))
    Ts = np.ones(2)
    view = make_store_view(samples_store, logLs_store, Ts)
    obs_vars = np.array([np.var(samples_store[:, :, 0])])

    for n_burnin_thin in (0, 7):
        n_use = view.store_size - n_burnin_thin
        autocorr_lim, autocorr_cut, est_var_auto = autocorr_helper(view, 0, n_burnin_thin)
        n_eff_auto = np.array([n_use * view.n_cold / (est_var_auto / autocorr_lim[0])])

        cov_cross_lim, _cov_cross_cut, _est_var_cross = get_crosscorr_sum(
            view,
            n_burnin_thin,
            0,
            autocorr_lim,
            autocorr_cut,
            obs_vars,
            n_eff_auto,
        )

        assert cov_cross_lim.shape == (1 + (n_use - 1) // 2,)


def test_corr_summary_uses_stored_sample_count_for_n_eff() -> None:
    rng = np.random.default_rng(1)
    samples_store = rng.standard_normal((128, 2, 1))
    logLs_store = np.zeros((128, 2))
    Ts = np.ones(2)
    view = make_store_view(samples_store, logLs_store, Ts)

    summary = CorrelationSummary(do_corr_summary=True, do_autocorr=True, do_cross=False)
    summary.obs_vars.append(np.array([np.var(samples_store[:, :, 0])]))
    summary.corr_summary(view, 0)

    autocorr_lim, _autocorr_cut, est_var_auto = autocorr_helper(view, 0, 0)
    expected_n_eff = view.store_size * view.n_cold / (est_var_auto / autocorr_lim[0])

    assert_allclose(summary.n_eff_preds_auto[-1][0], expected_n_eff)
    assert_allclose(summary.n_eff_preds[-1][0], expected_n_eff)
