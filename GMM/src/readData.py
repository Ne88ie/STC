__author__ = 'annie'

import numpy as np
import struct


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
    return np.array(struct.unpack('<{0}f'.format(row*col), file.read(4*row*col)), dtype=float).reshape(row, col)



def getFeatures(path):
    with open(path, 'rb') as f:
        dim = struct.unpack('<i', f.read(4))[0]
        t_features = struct.unpack('<i', f.read(4))[0]
        return readMatrix(f, t_features, dim)


def getUbm(path, preprocessing=None):
    with open(path, 'rb') as f:
        dim = struct.unpack('<i', f.read(4))[0]
        m_gauss = struct.unpack('<i', f.read(4))[0]
        weightGauss = np.array(struct.unpack('<{0}f'.format(m_gauss), f.read(4*m_gauss)), dtype=float)
        means = readMatrix(f, m_gauss, dim)
        covarianceMatrix = readMatrix(f, m_gauss, dim)
        return Ubm(dim, m_gauss, weightGauss, means, covarianceMatrix, preprocessing)
