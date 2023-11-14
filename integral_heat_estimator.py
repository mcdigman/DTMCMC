import numpy as np
from scipy.special import factorial

def cumulant_integrand(cumulants,betas):
    n_chain = betas.size
    integrand_left = np.zeros(n_chain-1)
    integrand_right = np.zeros(n_chain-1)
    integrand_avg = np.zeros(n_chain-1)
    
    for itrt in range(0,n_chain-1):
        for itrc in range(0,len(cumulants)-1):
            n = itrc + 1
            integrand_left[itrt] += (-1)**n/factorial(n)*(betas[itrt+1]-betas[itrt])**n*(betas[itrt]*cumulants[itrc+1][itrt] - (n - 1)*cumulants[itrc][itrt])

    for itrt in range(1,n_chain):
        for itrc in range(0,len(cumulants)-1):
            n = itrc + 1
            integrand_right[itrt-1] -= (-1)**n/factorial(n)*(betas[itrt-1]-betas[itrt])**n*(betas[itrt]*cumulants[itrc+1][itrt] - (n - 1)*cumulants[itrc][itrt])

    integrand_avg = (integrand_left + integrand_right)/2.

    return integrand_left,integrand_right,integrand_avg


def cumulant_heat_cap_interp(cumulants,betas_orig,betas_new):
    n_chain = betas_orig.size 
    assert np.all(np.diff(betas_orig)<0.)
    assert np.all(np.diff(betas_new)<0.)

    betas_orig = betas_orig[::-1]
    cumulants = cumulants[:,::-1]

    estim_left = np.zeros(betas_new.size)
    estim_right = np.zeros(betas_new.size)
    estim_center = np.zeros(betas_new.size)

    for itrt1 in range(betas_new.size):
        beta1 = betas_new[itrt1] 

        if beta1 == 0.:
            continue

        itrt_old = np.searchsorted(betas_orig,betas_new[itrt1]) - 1
        if itrt_old == n_chain:
            itrt_old = n_chain - 1
        if itrt_old < 0:
            itrt_old = 0

        beta0 = betas_orig[itrt_old]
        assert beta1 >= beta0
        estim_left[itrt1] = beta0**2*cumulants[1,itrt_old]

        #if len(cumulants) > 2:
        #    estim_left[itrt1] += (beta1-beta0)*(2*beta0*cumulants[1,itrt_old]+beta0**2*cumulants[2,itrt_old])

        for itrc in range(1,len(cumulants)-1):
            n = itrc 
            estim_left[itrt1] += (beta1-beta0)**n/factorial(n)*((n**2-n)*cumulants[itrc-1,itrt_old]+2*n*beta0*cumulants[itrc,itrt_old]+beta0**2*cumulants[itrc+1,itrt_old])
        #    estim_left[itrt1] -= (-1)**n/factorial(n-1)*(beta1-beta0)**(n-1)*(beta0*cumulants[itrc+1,itrt_old] - (n - 1)*cumulants[itrc,itrt_old])

        if itrt_old < n_chain - 1:
            beta0 = betas_orig[itrt_old+1]
            assert beta0 >= beta1
            #estim_right[itrt1] = beta0*cumulants[1,itrt_old+1]
            estim_right[itrt1] = beta0**2*cumulants[1,itrt_old+1]

            #if len(cumulants) > 2:
            #    estim_right[itrt1] += (beta1-beta0)*(2*beta0*cumulants[1,itrt_old+1]+beta0**2*cumulants[2,itrt_old+1])

            for itrc in range(1,len(cumulants)-1):
                n = itrc 
                estim_right[itrt1] += (beta1-beta0)**n/factorial(n)*((n**2-n)*cumulants[itrc-1,itrt_old+1]+2*n*beta0*cumulants[itrc,itrt_old+1]+beta0**2*cumulants[itrc+1,itrt_old+1])
            #for itrc in range(1,len(cumulants)-2):
            #    n = itrc + 1
            #    estim_right[itrt1] += (beta1-beta0)**(n-1)/factorial(n-1)*((n**2-n)*cumulants[itrc,itrt_old+1]+2*n*beta0*cumulants[itrc+1,itrt_old+1]+beta0**2*cumulants[itrc+2,itrt_old+1])
            #if len(cumulants) > 2:
            #    estim_right[itrt1] -= beta0*beta1*(beta1-beta0)*cumulants[2,itrt_old+1]
        #    for itrc in range(1,len(cumulants)-1):
        #        assert False
        #        n = itrc + 1
        #        estim_right[itrt1] -= (-1)**n/factorial(n-1)*(beta1-beta0)**(n-1)*(beta0*cumulants[itrc+1,itrt_old+1] - (n - 1)*cumulants[itrc,itrt_old+1])
            if itrt_old == 0:
                # exponentially cut off the right estimate in the beta=0 bin because it does not have the right asymptotics as beta->0
                estim_right[itrt1] = np.exp(-((1./beta0-1./beta1)*beta0)**2/10)*estim_right[itrt1]

            if estim_left[itrt1] != 0.:
                dbeta = betas_orig[itrt_old+1]-betas_orig[itrt_old]
                estim_center[itrt1] = (betas_orig[itrt_old+1]-beta1)/dbeta*estim_left[itrt1] + (beta1 - betas_orig[itrt_old])/dbeta*estim_right[itrt1]
            else:
                estim_center[itrt1] = estim_right[itrt1]
        else:
            estim_center[itrt1] = estim_left[itrt1]

    estim_left[estim_left<0.] = 0.
    estim_right[estim_right<0.] = 0.
    estim_center[estim_center<0.] = 0.

    return estim_left, estim_right, estim_center


