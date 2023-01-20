"""
Generates a Friedman dataset using the model:
s = 0.5/3
X ~ N(0,cov), where cov(i,j) = s*s*0.9**abs(i-j)

When X data are generated using this method:
- Each input variable has mean=0.5 and standard deviation=0.5/3.
- ~99 per cent of samples have values in [0,1].
- X_i and X_j have a correlation = 0.9**abs(i-j)

Y = 10*sin(pi*X[:,50]*X[:,100]) + 20*(X[:,150]-0.5)**2 + 10*X[:,200] + 5*X[:,250] + 0.1 * N(0,1)

"""

import numpy as np
from numpy.random import normal, multivariate_normal
import _pickle

nsamples_LS = 300
nsamples_VS = 100
nsamples_TS = 100
nfeat_irrelevant = 300
ndatasets = 10

nfeat_relevant = 5
nsamples = nsamples_LS + nsamples_VS + nsamples_TS
nfeat = nfeat_relevant+nfeat_irrelevant

m = np.zeros(nfeat) + 0.5
idx = np.arange(nfeat)
idx = np.tile(idx,(nfeat,1))
s = 0.5/3
cov = s*s*0.9**abs(idx-np.transpose(idx))


for n in range(ndatasets): 
    X = multivariate_normal(m,cov,size=nsamples)
    y = 10*np.sin(np.pi*X[:,50]*X[:,100]) + 20*(X[:,150]-0.5)**2 + 10*X[:,200] + 5*X[:,250] + 0.1*normal(size=nsamples)

    XLS = X[:nsamples_LS]
    XVS = X[nsamples_LS:nsamples_LS + nsamples_VS]
    XTS = X[nsamples_LS + nsamples_VS:]

    yLS = y[:nsamples_LS]
    yVS = y[nsamples_LS:nsamples_LS + nsamples_VS]
    yTS = y[nsamples_LS + nsamples_VS:]

    f = open('friedman_%d_LS.pkl' % (n + 1), 'wb')
    _pickle.dump((XLS, yLS), f)
    f.close()

    f = open('friedman_%d_VS.pkl' % (n + 1), 'wb')
    _pickle.dump((XVS, yVS), f)
    f.close()

    f = open('friedman_%d_TS.pkl' % (n + 1), 'wb')
    _pickle.dump((XTS, yTS), f)
    f.close()
    
    
