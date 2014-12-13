from __future__ import print_function, division
import numpy as np
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

def getGamma(ubm, features):
    print('\t\tStart getGamma')
    numberFeatures = features.shape[0]
    gamma = np.empty((numberFeatures, ubm.numberGauss))
    for t in xrange(numberFeatures):
        sys.stdout.write('\r' + '\t\t{0:.0%} of sample'.format((t + 1)/numberFeatures))
        sys.stdout.flush()
        x_t_numberGauss = np.array(features[t].tolist() * ubm.numberGauss).reshape(ubm.numberGauss, -1)
        x_t_numberGauss = x_t_numberGauss - ubm.means
        gamma[t] = np.exp(np.sum(-0.5 * x_t_numberGauss * x_t_numberGauss * ubm.covarianceMatrix, axis=1)) / ubm.sqrDetConv

    gamma *= ubm.weightGauss
    sumGammaOnGauss = np.sum(gamma)
    gamma /= sumGammaOnGauss
    print('\r\t\tFinish getGamma')
    return gamma


def getNewMeans(ubm, features, r):
    print('\tStart getNewMeans')
    ubm.sqrDetConv = np.multiply.reduce(np.power(ubm.covarianceMatrix, 0.5), axis=1)
    ubm.covarianceMatrix = np.power(ubm.covarianceMatrix, -1)
    gamma = getGamma(ubm, features)
    f_s = np.empty((ubm.means.shape))
    features = features.T
    for m in xrange(ubm.numberGauss):
        f_s[m] = np.sum(features * gamma[:, m], axis=1)
    n_plus_r = np.sum(gamma) + r
    newMeans = 1/n_plus_r * f_s + r/n_plus_r * ubm.means
    print('\tFinish getNewMeans')
    return newMeans
