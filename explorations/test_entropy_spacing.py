import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid

from DTMCMC.temperature_ladder_helpers import entropy_spaced_betas
from integ_box_filt import cumulants_from_Ts

Ts_in = np.load('Ts_cake_gold.npy')
cumulants_in = np.load('cumulants_cake_gold.npy')

betas_got, Ts_got = entropy_spaced_betas(32, 1, Ts_in, cumulants_in[1], n_inf_final=1, T_cold=1., correct_last=True)
cumulants_got = cumulants_from_Ts(Ts_got)

print(np.diff(cumulative_trapezoid(cumulants_got[1][::-1] * betas_got[::-1], betas_got[::-1], initial=0))[::-1])


integ_res = -cumulative_trapezoid(cumulants_got[1][::-1] * betas_got[::-1], betas_got[::-1], initial=0)[::-1]
integ_res -= integ_res[0]
plt.plot(integ_res)
plt.show()

plt.plot(np.diff(cumulative_trapezoid(cumulants_got[1][::-1] * betas_got[::-1], betas_got[::-1], initial=0))[::-1])
plt.show()
