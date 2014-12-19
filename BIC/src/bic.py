# coding=utf-8
from __future__ import print_function, division
import numpy as np
import time

__author__ = 'annie'

'''
dim = 20
t_features = 318045
'''

def print_state_of_process(whole):
    def my_decorator(func):
        def wrapped(*args, **kwargs):
            wrapped.time += 1
            print('\r\tCounting {0} {1:.2%}'.format(func.__name__, wrapped.time/whole), end='')
            if wrapped.time == whole:
                wrapped.time = 0
            return func(*args, **kwargs)
        wrapped.time = 0
        return wrapped
    return my_decorator


@print_state_of_process(31785) # whole = int((t_features - window_width + 1)/step + 1)
def delts_BIC(m1, m2, lam=3.05, use_diag=False):
    """lam=3.05 in 20091016-180000-RU-program-bics.csv 9305
    """
    n1 = m1.shape[0]
    n2 = m2.shape[0]
    n12 = n1 + n2
    dim = m1.shape[1]
    if use_diag:
        beta = dim
        bics = 0.5 * (n1 * np.prod(np.diag(np.cov(m1))) +
                      n2 * np.prod(np.diag(np.cov(m2))) -
                      n12 * np.prod(np.diag(np.cov(np.vstack((m1, m2))))) +
                      lam * beta * np.log(n12))

    else:
        beta = dim * (dim + 1)/2
        bics = 0.5 * (n1 * np.multiply(*np.linalg.slogdet(np.cov(m1))) +
                      n2 * np.multiply(*np.linalg.slogdet(np.cov(m2))) -
                      n12 * np.multiply(*np.linalg.slogdet(np.cov(np.vstack((m1, m2))))) +
                       lam * beta * np.log(n12))
    return bics


def get_ponts(features, window_width=200, step=10, lam=3.05, use_diag=False):
    """
    По массиву признаков пробегаем окном шириной window_width с шагом step. Окно разбиваем на две половинки. Для точки -
    середины окна - рассчитываем delts_BIC.
    :param features: список признаков, каждый из которых есть первые 20 коэффициентов MFCC (без С0)
    :param window_width: размер окна
    :param step: шаг
    :return:
    """
    start = time.time()
    mid = window_width/2

    # прибавляем нули слева и справа, чтобы потом было удобнее искать локальные максимумы
    d_bics = np.array([0] + [delts_BIC(features[i: i+mid], features[i+mid: i+window_width], lam, use_diag)
                       for i in range(0, features.shape[0] - (window_width - 1), step)] + [0])
    local_max = np.array([int(mid + step * (i-1)) for i in np.nonzero(d_bics > 0)[0] if np.all(d_bics[i-1: i+2: 2] < d_bics[i])])
    deltaTime = time.time() - start
    print('\r\tDone for {0:.0f}m {1:.0f}s'.format(deltaTime / 60, deltaTime % 60))
    return local_max


def get_recall(reference_points, getting_points, files_names, step):
    """
    Подсчитываем полноту для нескольких итераций. Мы считаем, что точка смены диктора point определена,
    если есть пересечние с интервлом [point-3*step : point-3*step] включительно.
    :param reference_points: список идеальных разметок
    :param getting_points: список полученных разметок
    :param files_names: имена файлов
    """
    recalls = []
    for n, reference in enumerate(reference_points):
        correct = 0
        for point in reference:
            interval_of_reference_points = np.array([point - i for i in range(1, 3*step+1)] + [point] + [point + i for i in range(1, 3*step+1)])
            if np.intersect1d(interval_of_reference_points, getting_points[n], assume_unique=True).size:
                correct += 1
        recalls.append(correct/reference.size)
    for name, rcl in zip(files_names, recalls):
        print('For {0} RECALL = {1:.2%}'.format(name, rcl))
    print('Average RECALL = {0:.2%}'.format(np.average(recalls)))
    return recalls


