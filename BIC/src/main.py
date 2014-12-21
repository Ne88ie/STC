# coding=utf-8
from __future__ import print_function, division
import os
import time
from IOData import get_features, get_points, save_points, Logger, Graph
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

    def search_points(path, graph, window_width=200, step=10, lam=3, use_diag=False, confidence_interval=50):
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
                local_max, graphs = find_points(features, window_width, step, lam, use_diag, confidence_interval)
                for g in graphs:
                    graph.add_plot(i-1, *g)
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

    def do_experiment(path, nomer, window_width=200, step=10, lam=3, use_diag=False, confidence_interval=50):
        """
        Проводим оценку поиска точек для всех файлов из path
        """
        start = time.time()
        try:
            if not os.path.exists(pathToTempDir):
                os.mkdir(pathToTempDir)
            graph = Graph(4)
            local_max_points, files_names = search_points(path, graph, window_width, step, lam, use_diag, confidence_interval)
            reference_points = get_referens_points(path)
            for i, ref in enumerate(reference_points):
                graph.add_plot(i, ref, [0]*len(ref), 'mo', 'ref_pints')
            recalls, precisions = get_recals_and_precisions(reference_points, local_max_points, files_names)
            logger.write(window_width, step, lam, use_diag, recalls, precisions, confidence_interval)
            graph.show(os.path.join(pathToTempDir, 'graph'+str(nomer+1)+'.png'))
        finally:
            deltaTime = time.time() - start
            print('Experiment done for {0:.0f}m {1:.0f}s\n'.format(deltaTime / 60, deltaTime % 60))

    def optimaze(path=pathToDevDir):
        """
        Выполняем вторую часть: ищем оптимальные параметры алгоритма.
        """
        parametrs = ((120, 15, 3, False, 450),) # (window_width, step, lam, use_diag, confidence_interval)

        start = time.time()
        try:
            for i, param in enumerate(parametrs):
                print('Experiment', i+1)
                do_experiment(path, i, *param)
        finally:
            deltaTime = time.time() - start
            print('Оptimization done for {0:.0f}m {1:.0f}s\n'.format(deltaTime / 60, deltaTime % 60))

        # For 20091016-180000-RU-program RECALL = 74.60%, PRECISION = 46.77%
        # For 20091019-040000-RU-program RECALL = 71.57%, PRECISION = 32.59%
        # For 20091019-050000-RU-program RECALL = 68.69%, PRECISION = 39.77%
        # For 20091019-080000-RU-program RECALL = 83.06%, PRECISION = 44.98%
        # Average RECALL = 74.48%, PRECISION = 41.02%

    def test(path=pathToTestDir):
        start = time.time()
        try:
            graph = Graph(4)
            local_max_points, files_names = search_points(path, graph, *(120, 15, 3, False, 450))
            for points, name  in zip(local_max_points, files_names):
                name = os.path.join(pathToTempDir, name + '.sys.pnts')
                save_points(name, points)
            graph.show(os.path.join(pathToTempDir, name +'.png'), show_ref=False)
        finally:
            deltaTime = time.time() - start
            print('Testing done for {0:.0f}m {1:.0f}s\n'.format(deltaTime / 60, deltaTime % 60))

    optimaze()
    test()

if __name__ == '__main__':
    main()

