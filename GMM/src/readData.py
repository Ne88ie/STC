__author__ = 'annie'

import numpy as np
import struct


class Ubm:
    def __init__(self, dim, m_gauss, weightGauss, means, covarianceMatrix):
        self.dim = dim
        self.m_gauss = m_gauss
        self.weightGauss = weightGauss
        self.means = means
        self.covarianceMatrix = covarianceMatrix

    def __str__(self):
        print('dim =', self.dim)
        print('m_gauss =', self.m_gauss)
        print('weightGauss =', self.weightGauss)
        print('means =', self.means)
        print('covarianceMatrix =', self.covarianceMatrix)


def readMatrix(file, row, col):
    firstRowMat = struct.unpack('<{0}f'.format(row), file.read(4*row))
    m = np.mat(firstRowMat).T
    for i in range(col-1):
        nestRowMat = struct.unpack('<{0}f'.format(row), file.read(4*row))
        a = np.mat(nestRowMat).T
        m = np.concatenate((m, a), 1)
    return m


def saveUbm(pathToFile, ubm, newMeans):
    with open(pathToFile, 'wb') as f:
        # f.write(ubm.dim.to_bytes(4, byteorder='little'))
        # f.write(ubm.m_gauss.to_bytes(4, byteorder='little'))
        f.write(struct.pack('<i', ubm.dim))
        f.write(struct.pack('<i', ubm.m_gauss))
        f.write(struct.pack('<%sf' % len(ubm.weightGauss), *ubm.weightGauss))
        for i in range(newMeans.shape[1]):
            f.write(struct.pack('<%sf' % newMeans.shape[0], *newMeans[:, i].T.tolist()[0]))
        for i in range(ubm.covarianceMatrix.shape[1]):
            f.write(struct.pack('<%sf' % ubm.covarianceMatrix.shape[0], *ubm.covarianceMatrix[:, i].T.tolist()[0]))


def getFeatures(path):
    with open(path, 'rb') as f:
        # dim = int.from_bytes(f.read(4), 'little')
        # t_features = int.from_bytes(f.read(4), 'little')
        dim = struct.unpack('<i', f.read(4))[0]
        t_features = struct.unpack('<i', f.read(4))[0]
        return readMatrix(f, t_features, dim)


def getUbm(path):
    with open(path, 'rb') as f:
        # m_gauss = int.from_bytes(f.read(4), 'little')
        dim = struct.unpack('<i', f.read(4))[0]
        m_gauss = struct.unpack('<i', f.read(4))[0]
        weightGauss = struct.unpack('<{0}f'.format(m_gauss), f.read(4*m_gauss))
        means = readMatrix(f, dim, m_gauss)
        covarianceMatrix = readMatrix(f, dim, m_gauss)
        return Ubm(dim, m_gauss, weightGauss, means, covarianceMatrix)


if __name__ == '__main__':
    pass