__author__ = 'annie'

from struct import pack, unpack
import numpy as np


class Ubm:
    def __init__(self, dim, m_gauss, weightGauss, means, covarianceMatrix, preprocessing=None):
        self.dim = dim
        self.numberGauss = m_gauss
        self.weightGauss = weightGauss
        self.means = means
        self.covMatDiag = covarianceMatrix
        self.sqrDetConv = None
        if preprocessing:
            self.__setSqrDetConvAndInvCov()

    def __setSqrDetConvAndInvCov(self):
        self.sqrDetConv = np.multiply.reduce(np.power(self.covMatDiag, 0.5), axis=1)
        self.covMatDiag = np.power(self.covMatDiag, -1)


def readMatrix(file, row, col):
    return np.array(unpack('<{0}f'.format(row*col), file.read(4*row*col)), dtype=float).reshape(row, col)



def getFeatures(path):
    with open(path, 'rb') as f:
        dim = unpack('<i', f.read(4))[0]
        t_features = unpack('<i', f.read(4))[0]
        return readMatrix(f, t_features, dim)


def getUbm(path, preprocessing=None):
    with open(path, 'rb') as f:
        dim = unpack('<i', f.read(4))[0]
        m_gauss = unpack('<i', f.read(4))[0]
        weightGauss = np.array(unpack('<{0}f'.format(m_gauss), f.read(4*m_gauss)), dtype=float)
        means = readMatrix(f, m_gauss, dim)
        covarianceMatrix = readMatrix(f, m_gauss, dim)
        return Ubm(dim, m_gauss, weightGauss, means, covarianceMatrix, preprocessing)


def saveUbm(pathToFile, pathToUbm, newMeans):
    with open(pathToFile, 'wb') as f:
        with open(pathToUbm, 'rb') as fromUbm:
            f.write(fromUbm.read(4*(newMeans.shape[0] + 2)))
            f.write(pack('<{0}f'.format(newMeans.size), *newMeans.ravel()))
            fromUbm.seek(4*newMeans.size, 1)
            f.write(fromUbm.read(4*newMeans.size))
    print('\tsaved ubm')