from time import perf_counter

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.special import gamma

cutoff = 10.
n_dim = 5

n_r = 1000
n_sim = 5000000

rs = np.linspace(0., np.sqrt(n_dim * cutoff**2), n_r)

integrand = rs.copy()**(n_dim - 1)


# integrand[rs>cutoff] = 2*rs[rs>cutoff]**(n_dim-1)*(np.pi-4*np.arccos(cutoff/rs[rs>cutoff]))
# integrand[rs>cutoff] = rs[rs>cutoff]**(n_dim-1)*(np.sqrt(n_dim)*cutoff-rs[rs>=cutoff])**(n_dim-1)
# integrand[rs>cutoff] *= cutoff**(n_dim-1)/(2*cutoff**(n_dim-1)*(np.pi-4*np.arccos(cutoff/cutoff)))
def areaFunc(rs):
    res = np.zeros(rs.size)
    r_high = rs[rs > cutoff]
    res[rs > cutoff] = 2**n_dim * cutoff * (r_high**2 - cutoff**2)**((n_dim - 1) / 2) + np.pi**(-n_dim / 2) * r_high**n_dim * (4 * np.pi**n_dim - 4**n_dim * np.pi * np.arccos(cutoff / r_high)**(n_dim - 1)) / (4 * gamma(1 + n_dim / 2))
    r_low = rs[rs <= cutoff]
    res[rs <= cutoff] = np.pi**(n_dim / 2) * r_low**(n_dim) / gamma(1 + n_dim / 2)
    return res


def rCutFunc(rs):
    return 2**n_dim * cutoff * (n_dim - 1) * rs * (rs**2 - cutoff**2)**((n_dim - 3) / 2) + np.pi**(n_dim / 2) / gamma(1 + n_dim / 2) * (-(4**(n_dim - 1) * cutoff * (n_dim - 1) * np.pi**(1 - n_dim) * rs * (rs * np.arccos(cutoff / rs))**(n_dim - 2)) / np.sqrt(rs**2 - cutoff**2) + n_dim * rs**(n_dim - 1) * (1 - (np.pi / 4)**(1 - n_dim) * np.arccos(cutoff / rs)**(n_dim - 1)))


integrand[rs > cutoff] = rCutFunc(rs[rs > cutoff]) / rCutFunc(cutoff + 1.e-13) * cutoff**(n_dim - 1)

res = 2 * np.pi / (np.pi / 2.) * cumulative_trapezoid(integrand, rs, initial=0)

print(res[-1], np.pi * rs[-1]**2, (2 * cutoff)**2, np.pi * cutoff**2)
print('answer: ', (2 * cutoff)**n_dim, 'result', res[-1], res[-1] / (2 * cutoff)**n_dim)

import matplotlib.pyplot as plt

r_lim = np.sqrt(n_dim) * cutoff
# cut_lower = cutoff/np.sqrt(n_dim)-0.01
# cut_low = cutoff/np.sqrt(n_dim)
# cut_high = cutoff/np.sqrt(n_dim)+0.01
# rs_sim_alt = np.random.uniform(cut_low**n_dim,cut_high**n_dim,n_sim)**(1/n_dim)
# simb1 = np.random.uniform(cut_low,cut_high,(n_sim,n_dim))
# rs_sim_alt = np.sqrt(np.sum(simb1**2,axis=1))
# rc = np.sum(rs_sim_alt>(np.sqrt(n_dim)*cut_low+(cut_high-cut_low)))/n_sim
#
# simb2 = np.random.uniform(cut_lower,cut_low,(n_sim,n_dim))
# rs_sim_alt2 = np.sqrt(np.sum(simb2**2,axis=1))
# rc2 = np.sum(rs_sim_alt2<(np.sqrt(n_dim)*cut_low-(cut_low-cut_lower)))/n_sim
# print(rc,rc2,(rc-rc2)/(2*0.01),rc2/rc)

t0 = perf_counter()
rs_sim = np.sqrt(np.sum(np.random.uniform(-cutoff, cutoff, (n_sim, n_dim))**2, axis=1))
bins_rec, bin_locs = np.histogram(rs_sim, 100000, range=(0., np.sqrt(n_dim) * cutoff))
bins_rec_orig = bins_rec.copy()
t1 = perf_counter()
print('finished init at ', t1 - t0)
max_itrb = 2000
for itrb in range(max_itrb):
    rs_sim = np.sqrt(np.sum(np.random.uniform(-cutoff, cutoff, (n_sim, n_dim))**2, axis=1))
    bins_loc, _ = np.histogram(rs_sim, 100000, range=(0., np.sqrt(n_dim) * cutoff))
    bins_rec += bins_loc
    tl = perf_counter()
    print('finished ', itrb, ' at ', tl - t1, 'project fin', (tl - t1) / (itrb + 1) * (max_itrb + 1))

plt.plot(bins_rec_prev2 / np.sum(bins_rec_prev))
plt.plot(bins_rec / np.sum(bins_rec))
plt.show()
bins_rec_prev = bins_rec.copy()
# bins2 = plt.hist(rs_sim_alt,1000,range=(0.,np.sqrt(n_dim)*cutoff),density=True)
plt.plot(rs, 1 / cutoff**n_dim * np.pi / 2. * integrand)
plt.show()

import sys

sys.exit()

rloc = bins[1][1:][bins[1][1:] >= cutoff]
dloc = bins[0][bins[1][1:] >= cutoff]
# plt.plot(rloc,dloc-np.pi/2*rloc**2/cutoff**3)
plt.plot(rloc, 8 * cutoff**3 * dloc - 4 * np.pi * rloc**2 + 2 * rloc**2 * np.arccos(cutoff / rloc))
plt.plot(rloc, 8 * cutoff**3 * dloc - 4 * np.pi * rloc**2)
plt.show()

# plt.plot(rs,areaFunc(rs)/(2*cutoff)**(n_dim))
# plt.plot(rs,np.full(n_r,1.))
# plt.plot(bins[1][1:],cumulative_trapezoid(bins[0],bins[1][1:],initial=0.))
# plt.show()

# plt.loglog(np.sqrt(n_dim)*cutoff-bins[1][1:][bins[1][1:]>=cutoff],(bins[0]/bins[1][1:]**(n_dim-1))[bins[1][1:]>=cutoff])
# plt.show()
