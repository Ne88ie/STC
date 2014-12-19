# coding=utf-8
from __future__ import print_function, division
import os
import time
from IOData import get_features, get_points, save_points, Logger
from bic import find_points, get_recals_and_precisions

__author__ = 'annie'

def main(start=1):
    """
    Реализация и иccледование байесовского информационного критерия (BIC, Besian Information Criterion), а также
    применение его к задаче поиска точек смены дикторов на фонограмме.
    """
    pathToData = '/Users/annie/SELabs/Kudashev/lab2/data'
    pathToDevDir = os.path.join(pathToData, 'dev')
    pathToTestDir = os.path.join(pathToData, 'test')
    pathToTempDir = os.path.join(pathToData, 'temp')
    pathToOptimizationFile = os.path.join(pathToTempDir, 'Optimization.csv')
    logger = Logger(pathToOptimizationFile, start)

    def searchPoints(path, window_width=200, step=10, lam=3.05, use_diag=False):
        """
        Выполняем первую часть: реализация алгоритма поиска точек смены дикторов на основе BIC.
        """
        i = 1
        local_max_points = []
        files_names = []
        for data in sorted(os.listdir(path)):
            if os.path.splitext(data)[-1] == '.features_bin':
                print('Search points in sample', i)
                features = get_features(os.path.join(path, data))
                local_max = find_points(features, window_width, step, lam, use_diag)
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
        Выполняем вторую часть: ищем оптимальные параметры алгоритма.
        """
        start = time.time()
        try:
            if not os.path.exists(pathToTempDir):
                os.mkdir(pathToTempDir)

            local_max_points, files_names = searchPoints(path, window_width, step, lam, use_diag)
            reference_points = get_referens_points(path)
            recalls, precisions = get_recals_and_precisions(reference_points, local_max_points, files_names, step)
            logger.write(window_width, step, lam, use_diag, recalls, precisions)
        finally:
            deltaTime = time.time() - start
            print('All done for {0:.0f}m {1:.0f}s\n'.format(deltaTime / 60, deltaTime % 60))


              # ((window_width, step, lam, use_diag),...)
    parametrs = (
                 (200, 10, 0.1, False),
                 (200, 10, 0.5, False),
                 (200, 10, 1, False),
                 (200, 10, 2, False),
                 (200, 10, 2.5, False),
                 (200, 10, 3, False),
                 (200, 10, 3.5, False),
                 (200, 10, 4, False),

                 (200, 10, 3.05, False),
                 (150, 10, 3.05, False),
                 (200, 10, 0.8, False),
                 (150, 10, 0.8, False),
                )
    for param in parametrs:
        optimaze(pathToDevDir, *param)

if __name__ == '__main__':
    main()

