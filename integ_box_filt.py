import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.ndimage import gaussian_filter

from cake_likelihood import get_loglike
from DTMCMC.temperature_ladder_helpers import Ts_to_betas, betas_to_Ts, entropy_spaced_betas
from moment_helpers import get_cumulants

n_dim = 5

rs = np.load('data/rs_rec_cake1.npy')
vals = np.load('data/box_rec_cake1.npy')

vals_integ = cumulative_trapezoid(vals, initial=0)
vals_integ_smooth = gaussian_filter(vals_integ, sigma=100)
vals_derive = np.gradient(vals_integ_smooth)

integrand = rs**(n_dim - 1)

vals_derive_norm = vals_derive / np.sum(vals_derive)

integrand_norm = integrand * vals_derive_norm[np.argmin(np.abs(rs - 10)) - 1] / integrand[np.argmin(np.abs(rs - 10))]

integ_stitch = np.hstack([integrand_norm[:np.argmin(np.abs(rs - 10))], vals_derive_norm[np.argmin(np.abs(rs - 10)) - 1:]])
integ_stitch = integ_stitch / np.trapezoid(integ_stitch, rs) * (2 * 10)**5

density_final = integ_stitch.copy()


loglikes = np.zeros(rs.size)

loglikes = loglikes


def get_density_pred(beta):
    density0 = np.trapezoid(density_final * np.exp(beta * (loglikes - loglikes[0])), rs)

    loglikes_correct = beta * (loglikes - loglikes[0]) - np.log(density0)

    density1 = np.trapezoid(density_final * np.exp(loglikes_correct), rs)
    assert np.isclose(density1, 1., atol=1.e-14, rtol=1.e-12)

    density_res = density_final * np.exp(loglikes_correct)
    return density_res


def cumulants_from_Ts(Ts):
    betas = Ts_to_betas(Ts)
    n_t = betas.size
    cumulants = np.zeros((6, n_t))

    for itrt in range(n_t):
        beta = betas[itrt]

        density_res = get_density_pred(beta)

        logL_powers = np.zeros(6)
        for itrp in range(logL_powers.size):
            logL_powers[itrp] = np.trapezoid(loglikes**(itrp + 1) * density_res, rs)

        cumulants[:, itrt] = get_cumulants(logL_powers)
    return cumulants


integrated_curve = cumulative_trapezoid(integ_stitch, rs, initial=0)
integrated_curve_interp = InterpolatedUnivariateSpline(rs, integrated_curve, k=3, ext=2)
ratio_got = integrated_curve_interp(10) / integrated_curve_interp(np.sqrt(5) * 10)
volume_in = (8 * np.pi**2 * 10**5 / 15)
volume_tot = (2 * 10)**5
ratio_pred = volume_in / volume_tot
print('ratio in-out', ratio_got, ratio_pred, ratio_got / ratio_pred)

assert np.isclose(np.trapezoid(density_final, rs), (2 * 10)**5, atol=1.e-14, rtol=1.e-12)
point_loc = np.zeros(n_dim)
for itrr in range(rs.size):
    point_loc[0] = rs[itrr]
    loglikes[itrr] = get_loglike(point_loc)


do_recalc = False
if do_recalc:
    import matplotlib.pyplot as plt
    Ts_in = np.load('data/Ts_cake_alternate1.npy')
    betas_in = Ts_to_betas(Ts_in)
    cumulants = cumulants_from_Ts(Ts_in)

# betas_geo, Ts_geo = geometric_spaced_betas(8192, 0, 1, 1.e-1, 1.e12, n_inf_final=1)
    betas_geo = np.linspace(1.3, 0, 8192)
    Ts_geo = betas_to_Ts(betas_geo)
# Ts_geo = np.hstack([np.linspace(0.8,1.e5,8191),np.inf])
# betas_geo = Ts_to_betas(Ts_geo)
    cumulants_geo = cumulants_from_Ts(Ts_geo)

    Ts_log = [Ts_geo]

    Ts_combine = Ts_geo.copy()
    cumulants_combine = cumulants_geo.copy()
    betas_combine = Ts_to_betas(Ts_combine)

    for itrb in range(3):
        betas_recalc, Ts_recalc = entropy_spaced_betas(8192, 0, Ts_combine, cumulants_combine[1], n_inf_final=1, T_cold=1., correct_last=True)
        Ts_log.append(Ts_recalc)
        cumulants_recalc = cumulants_from_Ts(Ts_recalc)
        Ts_combine = np.hstack([Ts_recalc, Ts_combine])
        cumulants_combine = np.hstack([cumulants_recalc, cumulants_combine])
        Ts_combine, argTs_combine = np.unique(Ts_combine, return_index=True)
        cumulants_combine = cumulants_combine[:, argTs_combine]
        betas_combine = Ts_to_betas(Ts_combine)

    Ts_combine = np.hstack([Ts_in, Ts_combine])
    cumulants_combine = np.hstack([cumulants, cumulants_combine])
    Ts_combine, argTs_combine = np.unique(Ts_combine, return_index=True)
    cumulants_combine = cumulants_combine[:, argTs_combine]
    betas_combine = Ts_to_betas(Ts_combine)

    cumulants_load = np.load('data/cumulants_cake_sequential1.npy')

    for itrb in range(len(Ts_log)):
        plt.semilogy(Ts_log[itrb])

    plt.show()

    plt.semilogx(Ts_in, cumulants[0] * betas_in**1)
    plt.semilogx(Ts_combine, cumulants_combine[0] * betas_combine**1)
    plt.semilogx(Ts_in, cumulants_load[0] * betas_in**1)
    plt.show()

    plt.semilogx(Ts_in, cumulants[1] * betas_in**2)
    plt.semilogx(Ts_combine, cumulants_combine[1] * betas_combine**2)
    plt.semilogx(Ts_in, cumulants_load[1] * betas_in**2)
    plt.show()

    plt.semilogx(Ts_in, cumulative_trapezoid(cumulants[1][::-1] * betas_in[::-1], betas_in[::-1], initial=0)[::-1])
    plt.semilogx(Ts_combine, cumulative_trapezoid(cumulants_combine[1][::-1] * betas_combine[::-1], betas_combine[::-1], initial=0)[::-1])
    plt.semilogx(Ts_in, cumulative_trapezoid(cumulants_load[1][::-1] * betas_in[::-1], betas_in[::-1], initial=0)[::-1])
    plt.show()

    plt.semilogx(Ts_in, cumulants[2] * betas_in**3)
    plt.semilogx(Ts_in, cumulants_load[2] * betas_in**3)
    plt.show()

    plt.semilogx(Ts_in, cumulants[3] * betas_in**4)
    plt.semilogx(Ts_in, cumulants_load[3] * betas_in**4)
    plt.show()

    plt.semilogx(Ts_in, cumulants[4] * betas_in**5)
    plt.semilogx(Ts_in, cumulants_load[4] * betas_in**5)
    plt.show()

    plt.semilogx(Ts_in, cumulants[5] * betas_in**6)
    plt.semilogx(Ts_in, cumulants_load[5] * betas_in**6)
    plt.show()

    entropy1 = np.trapezoid(cumulants[1] * betas_in, betas_in)
    entropy2 = np.trapezoid(cumulants_load[1] * betas_in, betas_in)
    print('entropy res', entropy1, entropy2, entropy2 - entropy1, entropy2 / entropy1 - 1.)


do_interpolant_quality_plots = False
if do_interpolant_quality_plots:
    plt.plot(vals_derive_norm)
    plt.plot(integ_stitch)
    plt.plot(integrand_norm[rs < 10.])
    plt.show()

    plt.plot(np.gradient(vals_derive_norm))
    plt.plot(np.gradient(integ_stitch))
    plt.plot(np.gradient(integrand_norm[rs < 10.]))
    plt.show()

    plt.plot(integrand_norm[rs < 10.][1:] - vals_derive_norm[rs[1:] < 10.])
    plt.plot(integ_stitch[rs < 10.][1:] - vals_derive_norm[rs[1:] < 10.])
    plt.show()


do_save = False
if do_save:
    cumulants1 = np.load('data/cumulants_cake_gold1.npy')
    Ts1 = np.load('data/Ts_cake_gold1.npy')

    cumulants2 = np.load('data/cumulants_cake_gold2.npy')
    Ts2 = np.load('data/Ts_cake_gold2.npy')

    cumulants3 = np.load('data/cumulants_cake_gold3.npy')
    Ts3 = np.load('data/Ts_cake_gold3.npy')

    cumulants4 = np.load('data/cumulants_cake_gold4.npy')
    Ts4 = np.load('data/Ts_cake_gold4.npy')

    Ts_full = np.hstack([Ts1, Ts2, Ts3, Ts4])
    cumulants_full = np.hstack([cumulants1, cumulants2, cumulants3, cumulants4])
    Ts_full, argTs_full = np.unique(Ts_full, return_index=True)
    cumulants_full = cumulants_full[:, argTs_full]
    betas_full = Ts_to_betas(Ts_full)
    betas_full, argbetas_full = np.unique(betas_full, return_index=True)
    betas_full = betas_full[::-1]
    argbetas_full = argbetas_full[::-1]
    Ts_full = Ts_full[argbetas_full]
    cumulants_full = cumulants_full[:, argbetas_full]

    np.save('data/cumulants_cake_gold.npy', cumulants_full)
    np.save('data/Ts_cake_gold.npy', Ts_full)
