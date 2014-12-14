# coding=utf-8
__author__ = 'annie'

from struct import pack, unpack
import numpy as np


class Ubm:
    def __init__(self, dim, m_gauss, weightGauss, means, covMatDiag, preprocessing=None):
        self.dim = dim
        self.numberGauss = m_gauss
        self.weightGauss = weightGauss
        self.means = means
        self.covMatDiag = covMatDiag
        if preprocessing:
            self.__setSqrDetConvAndInvCov()

    def __setSqrDetConvAndInvCov(self):
        """
        Вектор weightGauss поэлементно делим на вектор корней определителей ковариационных матриц.
        В covMatDiag сохраняем диагонали обратнных ковариационых матриц.
        """
        self.weightGauss /= np.prod(np.sqrt(self.covMatDiag), axis=1)
        self.covMatDiag = 1 / self.covMatDiag


def readMatrix(file, row, col):
    return np.array(unpack('<{0}f'.format(row*col), file.read(4*row*col)), dtype=np.float64).reshape(row, col)



def getFeatures(path):
    with open(path, 'rb') as f:
        dim = unpack('<i', f.read(4))[0]
        t_features = unpack('<i', f.read(4))[0]
        return readMatrix(f, t_features, dim)


def getUbm(path, preprocessing=None):
    with open(path, 'rb') as f:
        dim = unpack('<i', f.read(4))[0]
        m_gauss = unpack('<i', f.read(4))[0]
        weightGauss = np.array(unpack('<{0}f'.format(m_gauss), f.read(4*m_gauss)), dtype=np.float64)
        means = readMatrix(f, m_gauss, dim)
        covMatDiag = readMatrix(f, m_gauss, dim)
        return Ubm(dim, m_gauss, weightGauss, means, covMatDiag, preprocessing)


def saveUbm(pathToFile, pathToUbm, newMeans):
    with open(pathToFile, 'wb') as f:
        with open(pathToUbm, 'rb') as fromUbm:
            f.write(fromUbm.read(4*(newMeans.shape[0] + 2)))
            f.write(pack('<{0}f'.format(newMeans.size), *newMeans.ravel()))
            fromUbm.seek(4*newMeans.size, 1)
            f.write(fromUbm.read(4*newMeans.size))
    print('\tsaved ubm')