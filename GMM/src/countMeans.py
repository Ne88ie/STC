from __future__ import print_function, division
from math import pi
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
   M1 M2 M3 ... M512
D1
D2
...
D39
----------------------------

'''



def getWeightNorm(weightGauss, covarianceDiag, means, x_t):
    """
    or
    return weightGauss * scipy.stats.multivariate_normal(mean=means, cov=covarianceDiag).pdf(x_t)
    """
    # dim = covarianceDiag.shape[1]
    # pdf  = weightGauss/((2*pi)**(dim/2) * np.multiply.reduce(covarianceDiag)[0, 0]**.5)
    pdf = weightGauss/(np.multiply.reduce(covarianceDiag)[0, 0]**.5) # np.linalg.det
    covarianceMatrix = np.mat(np.diag(covarianceDiag.T.tolist()[0]))
    difference = x_t - means.T
    pdf *= np.exp(-0.5*np.multiply(difference, np.diag(np.linalg.inv(covarianceMatrix))) * difference.T)
    return pdf

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
    gamma = np.mat(np.empty((features.shape[0], ubm.means.shape[1])))
    for t in range(features.shape[0]):
        if (t + 1) % 500 == 0:
            sys.stdout.write('\r' + '\t\t{0:.0%} of t'.format((t + 1)/features.shape[0]))
            sys.stdout.flush()
        for gauss in range(ubm.means.shape[1]):
            gamma[t, gauss] = getWeightNorm(ubm.weightGauss[gauss],
                                            ubm.covarianceMatrix[:, gauss],
                                            ubm.means[:, gauss],
                                            features[t, :])
    for t in range(features.shape[0]):
        denominator = np.sum(gamma[t, :])
        for gauss in range(ubm.means.shape[1]):
            gamma[t, gauss] = gamma[t, gauss]/denominator if denominator else gamma[t, gauss]/0.0000001
    print('\r\t\tFinish getGamma')
    return gamma


def getNewMeans(ubm, features, r, nameModel):
    print('\tStart getNewMeans')
    newMeans = np.mat(np.empty(ubm.means.shape))
    gamma = getGamma(ubm, features)
    with open('../data/tempGamma/' + nameModel + '.db', 'wb') as gammaDB:
        pickle.dump(gamma, gammaDB)
    # with open('../data/tempGamma/' + nameModel + '.db', 'rb') as gammaDB:
    #     gamma = pickle.load(gammaDB)
    for gauss in range(ubm.means.shape[1]):
        sys.stdout.write('\r\tgauss {0}'.format(gauss + 1)); sys.stdout.flush()
        nGauss = np.sum(gamma[:, gauss])
        alpha = nGauss/(nGauss + r)
        fGauss = sum((features[t, :] * gamma[t, gauss] for t in range(features.shape[0])))/nGauss
        newMeans[:, gauss] = alpha * fGauss.T + (1 - alpha) * ubm.means[:, gauss]
    print('\r\tFinish getNewMeans')
    return newMeans
