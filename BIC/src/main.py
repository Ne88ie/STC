# coding=utf-8
from __future__ import print_function, division
import os
import time


from IOData import get_features, get_points, save_points, write_to_optimizatio_file
from bic import get_ponts, get_recall

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
    pathToOptimizationFile = os.path.join(pathToTempDir, 'Optimization.txt')
    logger = write_to_optimizatio_file(pathToOptimizationFile)

    def searchPoints(path, window_width=200, step=10, lam=3.05, use_diag=False):
        """
        Поиска точек смены дикторов на основе BIC.
        """
        i = 1
        local_max_points = []
        files_names = []
        for data in sorted(os.listdir(path)):
            if os.path.splitext(data)[-1] == '.features_bin':
                print('Features', i)
                features = get_features(os.path.join(path, data))
                local_max = get_ponts(features, window_width=200, step=10, lam=3.05, use_diag=False)
                local_max_points.append(local_max)
                files_names.append(os.path.splitext(data)[0])
                i += 1
        return local_max_points, files_names

    def get_referens_points(path):
        """
        Получаем файлы идеальной разметки.
        """
        reference_points = [get_points(os.path.join(path, name)) for name in sorted(os.listdir(path))
                            if os.path.splitext(name)[-1] == '.pnts' ]
        return reference_points

    def optimaze(path, window_width=200, step=10, lam=3.05, use_diag=False):
        """
        Выполняем первую часть: реализация алгоритма поиска точек смены дикторов на основе BIC.
        """
        start = time.time()
        try:
            if not os.path.exists(pathToTempDir):
                os.mkdir(pathToTempDir)

            local_max_points, files_names = searchPoints(path, window_width=200, step=10, lam=3.05, use_diag=False)
            reference_points = get_referens_points(path)
            recalls = get_recall(reference_points, local_max_points, files_names, step)
            logger(window_width, step, lam, use_diag, recalls)
        finally:
            deltaTime = time.time() - start
            print('All done for {0:.0f}m {1:.0f}s\n'.format(deltaTime / 60, deltaTime % 60))


              # ((window_width, step, lam, use_diag),...)
    parametrs = ((200, 10, 3.05, False),
                 (200, 10, 2, False),
                 (200, 10, 0.8, True),
                 (200, 10, 1.2, True),
                 (150, 10, 3.05, False),
                 (150, 10, 0.8, True))
    for param in parametrs:
        optimaze(pathToDevDir, *param)

if __name__ == '__main__':
    main()

