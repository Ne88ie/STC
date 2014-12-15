# coding=utf-8
from __future__ import print_function, division
import numpy as np

__author__ = 'annie'

'''
D = 39 размерность вектора признаков
M = 512 количество гаусойд
T1 ~ 4000-22000 количество признаков
____________________________
features_bin T * D
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
ubm.means M * D
----------------------------
   D1 D2 D3 ... D39
M1
M2
...
M39
----------------------------
____________________________
Gamma T * M
----------------------------
    M1 M2 M3 ... M512
T1
T2
T3
T4
...
TN
----------------------------
'''

def getGamma(gmm, features):
    """
    :param gmm: model is an instance of the class Gmm
    :param features: matrix T * D
    :return: matrix T * M without normalization
    """
    print('\tgetting gamma ...', end='')
    gamma = features[np.newaxis] - gmm.means[:, np.newaxis]
    gamma = -0.5 * np.sum(gamma * gamma * gmm.covMatDiag[:, np.newaxis], axis=2)
    gamma = np.exp(gamma).T * gmm.weightGauss
    print('\r' + ' '*22 + '\r', end='')
    return gamma


def getNewMeans(ubm, features, r=20):
    """
    :param ubm: universal bachground model - is an instance of the class Gmm
    :param features: matrix T * D
    :param r: relevance
    :return: the matrix of average values for speaker's model
    """
    gamma = getGamma(ubm, features)
    print('\tgetting new means ...', end='')
    gamma /= np.sum(gamma)
    n_plus_r = np.sum(gamma) + r

    # calculated first Baum–Welch's statistics
    statisticsFirst = np.empty((ubm.means.shape))
    featuresT = features.T
    for m in xrange(ubm.numberGauss):
        statisticsFirst[m] = np.sum(featuresT*gamma[:, m], axis=1)

    newMeans = 1 / n_plus_r * statisticsFirst + r / n_plus_r * ubm.means
    print('\r' + ' '*22 + '\r', end='')
    return newMeans

def getNewMeans3D(ubm, features, r=20):
    """
    :param ubm: universal bachground model - is an instance of the class Gmm
    :param features: matrix T * D
    :param r: relevance
    :return: the matrix of average values for speaker's model
    """
    gamma = getGamma(ubm, features)
    print('\tgetting new means ...', end='')
    gamma /= np.sum(gamma)
    n_plus_r = np.sum(gamma) + r

    # calculated first Baum–Welch's statistics
    statisticsFirst = np.sum(features[:, np.newaxis] * gamma[:, :, np.newaxis], axis=0)

    newMeans = 1 / n_plus_r * statisticsFirst + r / n_plus_r * ubm.means
    print('\r' + ' '*22 + '\r', end='')
    return newMeans


def criterionNeymanPearson(ubm, gmm, testFeatures):
    """
    :param ubm: universal bachground model - is an instance of the class Gmm
    :param gmm: speaker's model is an instance of the class Gmm
    :param testFeatures: matrix T * D
    :return: Neyman-Pearson's criterion
    """
    fGmmTest = np.sum(getGamma(gmm, testFeatures), axis=1)
    fUbmModel = np.sum(getGamma(ubm, testFeatures), axis=1)
    return sum(np.log(fGmmTest) - np.log(fUbmModel))/testFeatures.shape[0]
