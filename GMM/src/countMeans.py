from __future__ import print_function, division
import numpy as np
import pickle
import sys

__author__ = 'annie'

'''
T1 = 3991

D = 39
M = 512

____________________________
features_bin T*D
----------------------------
    D1 D2 D3 ... D39
T1
T2
T3
T4
...
T3991
----------------------------
____________________________
ubm D*M
----------------------------
   D1 D2 D3 ... D39
M1
M2
...
M39
----------------------------

'''

'''
Gamma T*M
    M1 M2 M3 ... M512
T1
T2
T3
T4
...
T3991

'''

def getF_GMM(ubm, features):
    sys.stdout.write(''); sys.stdout.flush()
    numberFeatures = features.shape[0]
    F_GMM = np.empty((numberFeatures, ubm.numberGauss))
    for t in xrange(numberFeatures):
        sys.stdout.write('\r' + '\t{0:.0%}'.format((t + 1)/numberFeatures))
        sys.stdout.flush()
        x_t_numberGauss = np.array(features[t].tolist() * ubm.numberGauss).reshape(ubm.numberGauss, -1)
        x_t_numberGauss = x_t_numberGauss - ubm.means
        F_GMM[t] = np.exp(np.sum(-0.5 * x_t_numberGauss * x_t_numberGauss * ubm.covarianceMatrix, axis=1)) / ubm.sqrDetConv
    F_GMM *= ubm.weightGauss
    sys.stdout.write('\r' + ' '*10 + '\r'); sys.stdout.flush()
    return F_GMM

def getGamma(ubm, features):
    gamma = getF_GMM(ubm, features)
    gamma /= np.sum(gamma)
    return gamma

def saveLogGMM(ubm, features, pathToLog):
    f_gmm = getF_GMM(ubm, features, pathToLog)
    with open(pathToLog, 'wb') as f:
        pickle.dump(np.log(np.sum(f_gmm, axis=1)), f)
    print('\tsaved log')

def saveUbm(pathToFile, pathToUbm, newMeans):
    with open(pathToFile, 'wb') as f:
        with open(pathToUbm, 'rb') as fromUbm:
            f.write(fromUbm.read(4*(newMeans.shape[0] + 2)))
            f.write(struct.pack('<{0}f'.format(newMeans.size), *newMeans.ravel()))
            fromUbm.seek(4*newMeans.size, 1)
            f.write(fromUbm.read(4*newMeans.size))
    print('\tsaved ubm') 

def getNewMeans(ubm, features, r):
    gamma = getGamma(ubm, features)
    f_s = np.empty((ubm.means.shape))
    features = features.T
    for m in xrange(ubm.numberGauss):
        f_s[m] = np.sum(features * gamma[:, m], axis=1)
    n_plus_r = np.sum(gamma) + r
    newMeans = 1/n_plus_r * f_s + r/n_plus_r * ubm.means
    return newMeans
