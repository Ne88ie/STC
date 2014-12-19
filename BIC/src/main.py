# coding=utf-8
from __future__ import print_function, division
import os
import time
import pickle

from IOData import get_features, get_points, save_points
from bic import run

__author__ = 'annie'

def main():
    """
    Реализация и иccледование байесовского информационного критерия (BIC, Besian Information Criterion), а также
    применение его к задаче поиска точек смены дикторов на фонограмме.
    """
    pathToData = '/Users/annie/SELabs/Kudashev/lab2/data'
    pathToDevDir = os.path.join(pathToData, 'dev')
    pathToTestDir = os.path.join(pathToData, 'test')
    pathToTempDir = os.path.join(pathToData, 'temp')

    def searchPoints(path=pathToDevDir):
        """
        Выполняем первую часть: реализация алгоритма поиска смены дикторов на основе BIC.
        """
        start = time.time()
        if not os.path.exists(pathToTempDir):
            os.mkdir(pathToTempDir)
        i = 1
        for data in sorted(os.listdir(path)):
            if os.path.splitext(data)[-1] == '.features_bin':
                print('Features', i)
                features = get_features(os.path.join(path, data))
                local_max = run(features)
                with open(os.path.join(pathToTempDir, os.path.splitext(data)[0] + '-bics.db'), 'wb') as f:
                    pickle.dump(local_max, f)
                with open(os.path.join(pathToTempDir, os.path.splitext(data)[0] + '-bics.csv'), 'w') as f:
                    f.write('\n'.join(map(str, local_max)))
                i += 1
            elif os.path.splitext(data)[-1] == '.pnts':
                points = get_points(os.path.join(path, data))
                with open(os.path.join(pathToTempDir, os.path.splitext(data)[0] + '.csv'), 'w') as f:
                    f.write('\n'.join(map(str, points)))
        deltaTime = time.time() - start
        print('All done for {0:.0f}m {1:.0f}s'.format(deltaTime / 60, deltaTime % 60))

    searchPoints()

if __name__ == '__main__':
    main()

