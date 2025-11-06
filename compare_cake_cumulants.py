import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumtrapz

import DTMCMC.temperature_ladder_helpers as th

Ts = np.load('Ts_cake_random1.npy')
betas = th.Ts_to_betas(Ts)

cumulants1 = np.load('cumulants_cake_random1.npy')
cumulants2 = np.load('cumulants_cake_sequential1.npy')
cumulants3 = np.load('cumulants_cake_adjacent1.npy')
cumulants4 = np.load('cumulants_cake_alternate1.npy')

plt.plot(cumulants1[0] * betas**1)
plt.plot(cumulants2[0] * betas**1)
plt.plot(cumulants3[0] * betas**1)
plt.plot(cumulants4[0] * betas**1)

plt.show()


plt.plot(cumulants1[1] * betas**2)
plt.plot(cumulants2[1] * betas**2)
plt.plot(cumulants3[1] * betas**2)
plt.plot(cumulants4[1] * betas**2)

plt.show()

plt.plot(cumulants1[2] * betas**3)
plt.plot(cumulants2[2] * betas**3)
plt.plot(cumulants3[2] * betas**3)
plt.plot(cumulants4[2] * betas**3)

plt.show()


plt.plot(cumulants1[3] * betas**4)
plt.plot(cumulants2[3] * betas**4)
plt.plot(cumulants3[3] * betas**4)
plt.plot(cumulants4[3] * betas**4)

plt.show()


plt.plot(cumulants1[4] * betas**5)
plt.plot(cumulants2[4] * betas**5)
plt.plot(cumulants3[4] * betas**5)
plt.plot(cumulants4[4] * betas**5)

plt.show()

plt.plot(cumulants1[5] * betas**6)
plt.plot(cumulants2[5] * betas**6)
plt.plot(cumulants3[5] * betas**6)
plt.plot(cumulants4[5] * betas**6)

plt.show()


plt.plot(cumtrapz(cumulants1[1][::-1] * betas[::-1], betas[::-1], initial=0.))
plt.plot(cumtrapz(cumulants2[1][::-1] * betas[::-1], betas[::-1], initial=0.))
plt.plot(cumtrapz(cumulants3[1][::-1] * betas[::-1], betas[::-1], initial=0.))
plt.plot(cumtrapz(cumulants4[1][::-1] * betas[::-1], betas[::-1], initial=0.))

plt.show()
