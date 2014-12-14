from __future__ import print_function, division
import numpy as np
import sys

__author__ = 'annie'

'''
D = 39
M = 512
T1 ~ 4000-7000

____________________________
features_bin T*D
----------------------------
    D1 D2 D3 ... D39
T1
T2
T3
T4
...
TN
----------------------------
____________________________
ubm M*D
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
TN

'''

def getF_GMM(ubm, features):
    numberFeatures = features.shape[0]
    F_GMM = np.concatenate((features[np.newaxis],)*ubm.numberGauss) - \
            np.concatenate((ubm.means[:, np.newaxis],)*numberFeatures, axis=1)
    F_GMM = -0.5 * np.sum(F_GMM * F_GMM * np.concatenate((ubm.covMatDiag[:, np.newaxis],)*numberFeatures, axis=1), axis=2)
    F_GMM = np.exp(F_GMM).T
    F_GMM *= ubm.weightGauss
    return F_GMM


def getGamma(ubm, features):
    gamma = getF_GMM(ubm, features)
    gamma /= np.sum(gamma)
    return gamma


def getLogGMM(ubm, features):
    print('\tGet LogGMM')
    f_gmm = getF_GMM(ubm, features)
    return np.log(np.sum(f_gmm, axis=1))


def getNewMeans(ubm, features, r=20):
    print('\tGet new means')
    gamma = getGamma(ubm, features)
    statisticsFirst = np.sum(np.concatenate((features[np.newaxis],)*ubm.numberGauss)
                      * np.concatenate((gamma.T[:, :, np.newaxis],)*ubm.dim, axis=2), axis=1)
    n_plus_r = np.sum(gamma) + r
    newMeans = 1 / n_plus_r * statisticsFirst + r / n_plus_r * ubm.means
    return newMeans


def criterionNeymanPearson(modelUbm, testFeatures):
    modelSample = getLogGMM(modelUbm, testFeatures)
    newMeans = getNewMeans(modelUbm, testFeatures, 20)
    modelUbm.means = newMeans
    testSample = getLogGMM(modelUbm, testFeatures)
    return sum(testSample - modelSample)/modelSample.size


