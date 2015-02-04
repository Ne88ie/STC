# coding=utf-8
from __future__ import print_function, division
__author__ = 'moiseeva'

import os
import time
import numpy as np
from struct import pack, unpack
from GMM.src.IOData import getGmm, getFeatures

PI = 0.5 # априорная вероятность начального состояния
A = np.array([[0.95, 0.05], [0.05, 0.95]]) # матрица переходов


class Viterbi:
    def __init__(self, path_to_features, pathes_to_gmms):
        self.features = getFeatures(path_to_features)
        self.gmms = []
        for path in pathes_to_gmms:
            self.gmms.append(getGmm(path))
        self.number_of_speakers = len(pathes_to_gmms) # K
        self.len = self.features.shape[0] # T
        self.dim = self.features.shape[1] # D
        self.sys = np.empty(self.len, dtype=np.int)

    def get_distribution_of_hidden_states(self, x_t):
        """
        Функция распределения скрытых состояний в виде GMM.
        :param x_t: 1 x dim
        :return:
        """
        res = []
        for gmm in self.gmms:
            gmm.weightGauss /= np.prod(np.sqrt(gmm.covMatDiag), axis=1)
            gmm.covMatDiag = 1 / gmm.covMatDiag
            buf = 0
            for i in range(gmm.numberGauss):
                dif = x_t - gmm.means[i]
                buf += gmm.weightGauss[i] * np.exp(np.sum(-0.5 * dif ** 2 * gmm.covMatDiag[i]))
            res.append(buf)
        return res

    def __decode(self, prevs, start):
        self.sys[-1] = start
        for i in range(self.len-1, 0, -1):
            self.sys[i-1] = prevs[i, self.sys[i]]

    def save_sys(self, path):
        with open(path, 'wb') as f:
            f.write(pack('<i', self.sys.size))
            f.write(pack('<{0}i'.format(self.sys.size), *self.sys)) # ravel?

    def run(self):
        v = np.empty((self.len, self.number_of_speakers), dtype=np.float) # T * K
        prevs = np.empty((self.len, self.number_of_speakers), dtype=np.int) # T * K
        v[0] = np.log(self.get_distribution_of_hidden_states(self.features[0])) + np.log(PI) # 1 * K
        print('... processing', end='')
        for t, features in enumerate(self.features[1:], start=1):
            v[t] = np.log(self.get_distribution_of_hidden_states(features))
            for k in range(self.number_of_speakers):
                jumps = (np.log(A[0, k]) + v[t-1, 0], np.log(A[1, k]) + v[t-1, 1])
                arg = int(np.argmax(jumps))
                v[t, k] += jumps[arg]
                prevs[t, k] = arg

            print('\r... processing {0:.2%}'.format((t+1.0)/self.len), end='')
        print('\r' + ' '*30 + '\r', end='')

        start = int(np.argmax(v[-1]))
        self.__decode(prevs, start)
        return self.sys

    @classmethod
    def get_indx(cls, path):
        with open(path, 'rb') as f:
            amt = unpack('<i', f.read(4))[0]
            return np.array(unpack('<{0}i'.format(amt), f.read(4*amt)), dtype=np.int)

    @classmethod
    def check(cls, path_to_ref, sys):
        if type(sys) == str:
            sys = cls.get_indx(sys)
        ref = cls.get_indx(path_to_ref)
        tp = np.count_nonzero(np.bitwise_xor(ref, sys) < 1)
        prc = tp/(sys.size or 0.0001)
        return prc

def test(path_to_dir, path_to_dir_sys):
    """
    :param path_to_dir: path to input data.
    :return:
    START
    File: fabrl, PRC = 72.54%
    File: fafop, PRC = 97.58%
    File: fafgm, PRC = 84.23%
    File: faawt, PRC = 89.64%
    File: faawu, PRC = 73.41%
    File: fadxy, PRC = 85.05%
    File: faawn, PRC = 85.38%
    File: faawe, PRC = 79.92%
    File: fadxv, PRC = 91.22%
    Average PRC = 84.33%
    EXECUTED FOR 73m 33s
    """
    start = time.time()
    print('START')
    try:
        tests = set()
        for file in os.listdir(path_to_dir):
            name = os.path.splitext(file)
            if name[-1] == '.features_bin':
                tests.add(name[0])
        prc_all = 0
        files = 0
        for name in tests:
            path = os.path.join(path_to_dir, name)
            path_to_features = path + '.features_bin'
            pathes_to_gmms = [path + '.1.gmm', path + '.2.gmm']
            path_to_ref = path + '.ref.indx'
            path_to_sys = os.path.join(path_to_dir_sys, name + '.sys.indx')
            vit = Viterbi(path_to_features, pathes_to_gmms)
            vit.run()
            vit.save_sys(path_to_sys)
            prc = vit.check(path_to_ref, path_to_sys)
            prc_all += prc
            files += 1
            print('File: {0}, PRC = {1:.2%}'.format(name, prc))
        print('Average PRC = {0:.2%}'.format(prc_all/files))
    finally:
        deltaTime = time.time() - start
        print('EXECUTED FOR {0:.0f}m {1:.0f}s\n'.format(deltaTime / 60, deltaTime % 60))


def check(dir_ref, dir_sys):
    prc_all = 0
    files = 0
    for file in os.listdir(dir_sys):
        name = file.split('.')
        if name[-1] == 'indx':
            path_to_ref = os.path.join(dir_ref, name[0] + '.ref.indx')
            path_to_sys = os.path.join(dir_sys, name[0] + '.sys.indx')
            prc = Viterbi.check(path_to_ref, path_to_sys)
            prc_all += prc
            files += 1
            print('File: {0}, PRC = {1:.2%}'.format(name[0], prc))
    print('Average PRC = {0:.2%}'.format(prc_all/files))


if __name__ == '__main__':
    path_to_base = '/Users/annie/SELabs/Kudashev/lab3/base'
    path_to_dir_sys = '/Users/annie/SELabs/Kudashev/lab3/res'
    test(path_to_base, path_to_dir_sys)
    # check(path_to_base, path_to_dir_sys)
