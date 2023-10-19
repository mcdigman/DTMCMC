import numpy as np

T0 = 1.
dense0 = np.array([100.,10.,1.])
dense0 /= np.sum(dense0)

n_bin = dense0.size

T1 = 10.
dense1 = dense0**(T0/T1)
dense1 /= np.sum(dense1)

n_run = 10000
n_draw = 100

infer0_res = np.zeros((n_run,n_bin))
infer1_res = np.zeros((n_run,n_bin))
infer01_res = np.zeros((n_run,n_bin))
infer10_res = np.zeros((n_run,n_bin))

for itrm in range(0,n_run):
    draws0 = np.histogram(np.random.choice(np.arange(0,n_bin),n_draw,p=dense0),bins=np.arange(0,n_bin+1))[0]
    draws1 = np.histogram(np.random.choice(np.arange(0,n_bin),n_draw,p=dense1),bins=np.arange(0,n_bin+1))[0]

    infer0_res[itrm] = draws0/np.sum(draws0)
    infer1_res[itrm] = draws1/np.sum(draws1)

    infer01_res[itrm] = draws1**(T1/T0)/np.sum(draws1**(T1/T0))#/n_draw
    #infer01_res[itrm] /= n_draw#np.sum(infer01_res[itrm])
    #infer01_res[itrm] *= np.sum(draws1**(T1/T0))/n_draw

    infer10_res[itrm] = infer0_res[itrm]**(T0/T1)
    infer10_res[itrm] /= np.sum(infer10_res[itrm])

print('got0',infer0_res.mean(axis=0))
print('loc0',infer0_res.mean(axis=0)/dense0)
print('inf0',infer01_res.mean(axis=0)/dense0)
print('got1',infer1_res.mean(axis=0))
print('loc1',infer1_res.mean(axis=0)/dense1)
print('inf1',infer10_res.mean(axis=0)/dense1)
print('std0',infer0_res.std(axis=0))
print('std1',infer1_res.std(axis=0))
