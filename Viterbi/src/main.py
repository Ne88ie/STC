__author__ = 'moiseeva'

import os
import time
import numpy as np
from struct import pack, unpack
from GMM.src.IOData import getGmm, getFeatures

PI = (0.5, 0.5) # априорные вероятности начального состояния
A = np.array([[0.95, 0.05], [0.05, 0.95]]) # матрица переходов


class Viterbi:
    def __init__(self, path_to_features, pathes_to_gmms):
        self.features = getFeatures(path_to_features)
        self.gmms = []
        for path in pathes_to_gmms:
            self.gmms.append(getGmm(path))
        self.number_of_speakers = len(pathes_to_gmms)
        self.len = self.features.shape[0]
        self.dim = self.features.shape[1]
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
                buf += gmm.weightGauss[i] * np.exp(np.sum(-0.5 * np.power(x_t - gmm.means[i], 2) * gmm.covMatDiag[i]))
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
        v = np.empty((self.len, self.number_of_speakers), dtype=np.float)
        prevs = np.empty((self.len, self.number_of_speakers), dtype=np.int)
        v[0] = self.get_distribution_of_hidden_states(self.features[0])
        v[0, 0] *= PI[0]
        v[0, 1] *= PI[1]
        print('\tprocessing', end='')
        for t, features in enumerate(self.features[1:], start=1):
            v[t] = self.get_distribution_of_hidden_states(features)

            # jumps = (A[0, 0]*v[t-1, 0], A[1, 0]*v[t-1, 1])
            # arg = int(np.argmax(jumps))
            # v[t, 0] = v[t, 0] * jumps[arg]
            # prevs[t, 0] = arg
            #
            # jumps = (A[0, 1]*v[t-1, 0], A[1, 1]*v[t-1, 1])
            # arg = int(np.argmax(jumps))
            # v[t, 1] = v[t, 1] * jumps[arg]
            # prevs[t, 1] = arg

            for k in range(self.number_of_speakers):
                jumps = (A[0, k]*v[t-1, 0], A[1, k]*v[t-1, 1])
                arg = int(np.argmax(jumps))
                v[t, k] *= jumps[arg]
                prevs[t, k] = arg

            print('\r\tprocessing {0:.2%}'.format((t+1)/self.len), end='')
        print('\r' + ' '*30 + '\r', end='')

        start = int(np.argmax(v[-1]))
        self.__decode(prevs, start)
        return self.sys

    @classmethod
    def get_ref(cls, path):
        with open(path, 'rb') as f:
            amt = unpack('<i', f.read(4))[0]
            return np.array(unpack('<{0}i'.format(amt), f.read(4*amt)), dtype=np.int)

    @classmethod
    def check(cls, path_to_ref, path_to_sys=None, sys=None):
        if not sys:
            sys = cls.get_ref(path_to_sys)
        ref = cls.get_ref(path_to_ref)
        tp = np.count_nonzero(np.bitwise_xor(ref, sys) < 1)
        print('PRC = {0:.2%}'.format(tp/sys.size))

def test(path_to_dir):
    start = time.time()
    print('START')
    try:
        tests = set()
        for file in os.listdir(path_to_dir):
            name = os.path.splitext(file)
            if name[-1] == '.features_bin':
                tests.add(name[0])
        for name in tests:
            print('File:', name)
            path = os.path.join(path_to_dir, name)
            path_to_features = path + '.features_bin'
            pathes_to_gmms = [path + '.1.gmm', path + '.2.gmm']
            path_to_ref = path + '.ref.indx'
            path_to_sys = os.path.join('C:/Users/moiseeva/PycharmProjects/data_kud_lab3/res', name + '.sys.indx')
            vit = Viterbi(path_to_features, pathes_to_gmms)
            vit.run()
            vit.save_sys(path_to_sys)
            vit.check(path_to_ref, path_to_sys)
    finally:
        deltaTime = time.time() - start
        print('EXECUTED FOR {0:.0f}m {1:.0f}s\n'.format(deltaTime / 60, deltaTime % 60))


if __name__ == '__main__':
    path_to_dir = 'C:/Users/moiseeva/PycharmProjects/data_kud_lab3/base/'
    test(path_to_dir)