import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.integrate import cumtrapz
from scipy.interpolate import InterpolatedUnivariateSpline
from cake_likelihood import get_loglike
from moment_helpers import get_cumulants
from DTMCMC.temperature_ladder_helpers import Ts_to_betas

n_dim = 5

rs = np.load('rs_rec_cake1.npy')
vals = np.load('box_rec_cake1.npy')

vals_integ = cumtrapz(vals,initial=0.)
vals_integ_smooth = gaussian_filter(vals_integ,sigma=100)
vals_derive = np.gradient(vals_integ_smooth)

integrand = rs**(n_dim-1)

vals_derive_norm = vals_derive/np.sum(vals_derive)

integrand_norm = integrand*vals_derive_norm[np.argmin(np.abs(rs-10))-1]/integrand[np.argmin(np.abs(rs-10))]

integ_stitch = np.hstack([integrand_norm[:np.argmin(np.abs(rs-10))],vals_derive_norm[np.argmin(np.abs(rs-10))-1:]])
integ_stitch = integ_stitch/np.trapz(integ_stitch,rs)*(2*10)**5

integrated_curve = cumtrapz(integ_stitch,rs,initial=0.)
integrated_curve_interp = InterpolatedUnivariateSpline(rs,integrated_curve,k=3,ext=2)
ratio_got = integrated_curve_interp(10)/integrated_curve_interp(np.sqrt(5)*10)
volume_in = (8*np.pi**2*10**5/15)
volume_tot = (2*10)**5
ratio_pred = volume_in/volume_tot
print('ratio in-out',ratio_got,ratio_pred,ratio_got/ratio_pred)

density_final = integ_stitch.copy()

assert np.isclose(np.trapz(density_final,rs),(2*10)**5,atol=1.e-14,rtol=1.e-12)


import matplotlib.pyplot as plt

loglikes = np.zeros(rs.size) 
point_loc = np.zeros(n_dim)
for itrr in range(rs.size):
    point_loc[0] = rs[itrr] 
    loglikes[itrr] = get_loglike(point_loc)


loglikes = loglikes

Ts = np.load('Ts_cake_alternate1.npy')
betas = Ts_to_betas(Ts)
n_t = betas.size
cumulants = np.zeros((6,n_t))

for itrt in range(n_t):
    beta = betas[itrt]

    density0 = np.trapz(density_final*np.exp(beta*(loglikes-loglikes[0])),rs)

    loglikes_correct = beta*(loglikes-loglikes[0])-np.log(density0)

    density1 = np.trapz(density_final*np.exp(loglikes_correct),rs)
    assert np.isclose(density1,1.,atol=1.e-14,rtol=1.e-12)

    logL_powers = np.zeros(6)
    for itrp in range(logL_powers.size):
        logL_powers[itrp] = np.trapz(loglikes**(itrp+1)*density_final*np.exp(loglikes_correct),rs)

    cumulants[:,itrt] = get_cumulants(logL_powers)

print(cumulants)
cumulants_load = np.load('cumulants_cake_sequential1.npy')

plt.plot(cumulants[0]*betas**1)
plt.plot(cumulants_load[0]*betas**1)
plt.show()

plt.plot(cumulants[1]*betas**2)
plt.plot(cumulants_load[1]*betas**2)
plt.show()


plt.plot(cumulants[2]*betas**3)
plt.plot(cumulants_load[2]*betas**3)
plt.show()

plt.plot(cumulants[3]*betas**4)
plt.plot(cumulants_load[3]*betas**4)
plt.show()

plt.plot(cumulants[4]*betas**5)
plt.plot(cumulants_load[4]*betas**5)
plt.show()

plt.plot(cumulants[5]*betas**6)
plt.plot(cumulants_load[5]*betas**6)
plt.show()

entropy1 = np.trapz(cumulants[1]*betas,betas)
entropy2 = np.trapz(cumulants_load[1]*betas,betas)
print('entropy res',entropy1,entropy2,entropy2-entropy1,entropy2/entropy1-1.)



do_interpolant_quality_plots = False
if do_interpolant_quality_plots:
    plt.plot(vals_derive_norm)
    plt.plot(integ_stitch)
    plt.plot(integrand_norm[rs<10.])
    plt.show()

    plt.plot(np.gradient(vals_derive_norm))
    plt.plot(np.gradient(integ_stitch))
    plt.plot(np.gradient(integrand_norm[rs<10.]))
    plt.show()

    plt.plot(integrand_norm[rs<10.][1:]-vals_derive_norm[rs[1:]<10.])
    plt.plot(integ_stitch[rs<10.][1:]-vals_derive_norm[rs[1:]<10.])
    plt.show()
