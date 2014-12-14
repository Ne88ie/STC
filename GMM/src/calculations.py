# coding=utf-8
from __future__ import print_function, division
import numpy as np

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
    F_GMM = features[np.newaxis] - ubm.means[:, np.newaxis]
    F_GMM = -0.5 * np.sum(F_GMM * F_GMM * ubm.covMatDiag[:, np.newaxis], axis=2)
    F_GMM = np.exp(F_GMM).T
    F_GMM *= ubm.weightGauss
    return F_GMM


def getGamma(ubm, features):
    gamma = getF_GMM(ubm, features)
    gamma /= np.sum(gamma)
    return gamma


def getLogGMM(ubm, features):
    print('\tgetting LogGMM ...', end='')
    f_gmm = getF_GMM(ubm, features)
    print('\r' + ' '*19 + '\r', end='')
    return np.log(np.sum(f_gmm, axis=1))


def getNewMeans(ubm, features, r=20):
    print('\tgetting new means ...', end='')
    gamma = getGamma(ubm, features)
    n_plus_r = np.sum(gamma) + r

    # вычисляем первую статисику Баумана-Уэлша
    statisticsFirst = np.empty((ubm.means.shape))
    featuresT = features.T
    for m in xrange(ubm.numberGauss):
        statisticsFirst[m] = np.sum(featuresT*gamma[:, m], axis=1)

    newMeans = 1 / n_plus_r * statisticsFirst + r / n_plus_r * ubm.means
    print('\r' + ' '*22 + '\r', end='')
    return newMeans

def getNewMeans3D(ubm, features, r=20):
    print('\tgetting new means ...', end='')
    gamma = getGamma(ubm, features)
    n_plus_r = np.sum(gamma) + r

    # вычисляем первую статисику Баумана-Уэлша
    statisticsFirst = np.sum(features[:, np.newaxis] * gamma[:, :, np.newaxis], axis=0)

    newMeans = 1 / n_plus_r * statisticsFirst + r / n_plus_r * ubm.means
    print('\r' + ' '*22 + '\r', end='')
    return newMeans


def criterionNeymanPearson(modelUbm, testFeatures, r=20):
    modelSample = getLogGMM(modelUbm, testFeatures)
    newMeans = getNewMeans3D(modelUbm, testFeatures, r)
    modelUbm.means = newMeans
    testSample = getLogGMM(modelUbm, testFeatures)
    return sum(testSample - modelSample)/modelSample.size


