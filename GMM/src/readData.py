__author__ = 'annie'

import numpy as np
import struct


class Ubm:
    def __init__(self, dim, m_gauss, weightGauss, means, covarianceMatrix):
        self.dim = dim
        self.numberGauss = m_gauss
        self.weightGauss = weightGauss
        self.means = means
        self.covarianceMatrix = covarianceMatrix
        self.sqrDetConv = None


def readMatrix(file, row, col):
    return np.array(struct.unpack('<{0}f'.format(row*col), file.read(4*row*col)), dtype=float).reshape(row, col)


def saveUbm(pathToFile, pathToUbm, newMeans):
    with open(pathToFile, 'wb') as f:
        with open(pathToUbm, 'rb') as fromUbm:
            f.write(fromUbm.read(4*(newMeans.shape[0] + 2)))
            f.write(struct.pack('<{0}f'.format(newMeans.size), *newMeans.ravel()))
            fromUbm.seek(4*newMeans.size, 1)
            f.write(fromUbm.read(4*newMeans.size))


def getFeatures(path):
    with open(path, 'rb') as f:
        dim = struct.unpack('<i', f.read(4))[0]
        t_features = struct.unpack('<i', f.read(4))[0]
        return readMatrix(f, t_features, dim)


def getUbm(path):
    with open(path, 'rb') as f:
        dim = struct.unpack('<i', f.read(4))[0]
        m_gauss = struct.unpack('<i', f.read(4))[0]
        weightGauss = np.array(struct.unpack('<{0}f'.format(m_gauss), f.read(4*m_gauss)), dtype=float)
        means = readMatrix(f, m_gauss, dim)
        covarianceMatrix = readMatrix(f, m_gauss, dim)
        return Ubm(dim, m_gauss, weightGauss, means, covarianceMatrix)
