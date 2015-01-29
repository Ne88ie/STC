# coding=utf-8
from __future__ import print_function, division
import numpy as np

__author__ = 'annie'

'''
D = 39 размерность вектора признаков
M = 512 количество гаусойд
T1 ~ 4000-22000 количество признаков

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

ubm.means M * D
----------------------------
   D1 D2 D3 ... D39
M1
M2
...
M39
----------------------------

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
    t_features = features.shape[0]
    gamma = np.empty((t_features, gmm.numberGauss)) # T * M
    for t in range(t_features):
        scope = -0.5 * ((features[t][np.newaxis, :] - gmm.means) ** 2 * gmm.covMatDiag).sum(axis=1)
        gamma[t] = gmm.weightGauss * np.exp(scope)
    return gamma


def getNewMeans(ubm, features, r=20):
    """
    :param ubm: universal bachground model - is an instance of the class Gmm
    :param features: matrix T * D
    :param r: relevance
    :return: the matrix of average values for speaker's model
    """
    gamma = getGamma(ubm, features) # T * M
    print('\tgetting new means ...', end='')
    n = gamma.sum(axis=0) # M
    a = n/(n+r) # M

    # calculated first Baum–Welch's statistics
    t_features = features.shape[0]
    statisticsFirst = np.zeros((ubm.numberGauss, ubm.dim)) # M * D
    for t in range(t_features):
        statisticsFirst += gamma[t][:, np.newaxis] * features[t][np.newaxis, :] # M * D
    statisticsFirst /= n[:, np.newaxis]

    newMeans = a[:, np.newaxis] * statisticsFirst + (1 - a)[:, np.newaxis] * ubm.means
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
