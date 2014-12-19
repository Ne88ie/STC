# coding=utf-8
from __future__ import print_function, division
import numpy as np
import time

__author__ = 'annie'

'''
dim = 20
t_features = 318045
'''


def print_state_of_process(whole=None):
    def my_decorator(func):
        def wrapped(*args, **kwargs):
            wrapped.time += 1
            if whole:
                print('\r\tCounting {0} {1:.2%}'.format(func.__name__, wrapped.time/whole), end='')
            else:
                print('\r\tCounting {0} {1}'.format(func.__name__, wrapped.time), end='')
            if wrapped.time == whole:
                wrapped.time = 0
            return func(*args, **kwargs)
        wrapped.time = 0
        return wrapped
    return my_decorator


@print_state_of_process(31785) # whole = int((t_features - window_width + 1)/step + 1)
def _delts_BIC(m1, m2, lam=3.05, use_diag=False):
    """
    Вычисляем дельта BIC для левого m1 и правого m2 интервалов
    :param m1:
    :param m2:
    :param lam: паоаметр лямбда
    :param use_diag: использовать ли только диагональ ковариационной матрицы
    :return:
    """
    n1 = m1.shape[0]
    n2 = m2.shape[0]
    n12 = n1 + n2
    dim = m1.shape[1]
    if use_diag:
        beta = dim
        bics = 0.5 * (n1 * np.sum(np.log(np.diag(np.cov(m1)))) +
                      n2 * np.sum(np.log(np.diag(np.cov(m2)))) -
                      n12 * np.sum(np.log(np.diag(np.cov(np.vstack((m1, m2)))))) +
                      lam * beta * np.log(n12))
    else:
        beta = dim * (dim + 1)/2
        bics = 0.5 * (n1 * np.multiply(*np.linalg.slogdet(np.cov(m1))) +
                      n2 * np.multiply(*np.linalg.slogdet(np.cov(m2))) -
                      n12 * np.multiply(*np.linalg.slogdet(np.cov(np.vstack((m1, m2))))) +
                       lam * beta * np.log(n12))
    return bics


def find_points(features, window_width=200, step=10, lam=3.05, use_diag=False):
    """
    По массиву признаков пробегаем окном шириной window_width с шагом step. Окно разбиваем на две половинки. Для точки -
    середины окна - рассчитываем _delts_BIC.
    :param features: список признаков, каждый из которых есть первые 20 коэффициентов MFCC (без С0)
    :param window_width: размер окна
    :param step: шаг
    :return: массив локальных максимумов
    """
    start = time.time()
    mid = window_width/2

    # прибавляем нули слева и справа, чтобы потом было удобнее искать локальные максимумы
    d_bics = np.array([0] + [_delts_BIC(features[i: i+mid], features[i+mid: i+window_width], lam, use_diag)
                       for i in range(0, features.shape[0] - (window_width - 1), step)] + [0])
    local_max = np.array([int(mid + step * (i-1)) for i in np.nonzero(d_bics < 0)[0] if np.all(d_bics[i-1: i+2: 2] > d_bics[i])])
    deltaTime = time.time() - start
    print('\r\tDone for {0:.0f}m {1:.0f}s'.format(deltaTime / 60, deltaTime % 60))
    return local_max


@print_state_of_process()
def _recal_and_precision(reference, getting, step=10):
    """
    Подсчитывае полноту. Мы считаем, что точка смены диктора point определена,
    если есть пересечение с интервалом [point-3*step : point-3*step] включительно.
    :param reference:
    :param getting:
    :param step:
    :return:
    """
    correct = 0
    for point in reference:
        interval_of_reference_points = np.array([point - i for i in range(1, 3*step+1)] + [point] + [point + i for i in range(1, 3*step+1)])
        if np.intersect1d(interval_of_reference_points, getting, assume_unique=True).size:
            correct += 1
    return correct/reference.size, correct/getting.size


def get_recals_and_precisions(reference_points, getting_points, files_names, step):
    """
    Подсчитываем полноту для нескольких итераций.
    :param reference_points: список идеальных разметок
    :param getting_points: список полученных разметок
    :param files_names: имена файлов
    :return: список recalls
    """
    start = time.time()
    print('Get recalls')
    recal_and_precision = [_recal_and_precision(ref, get, step) for ref, get in zip(reference_points, getting_points)]
    recalls, precisions = zip(*recal_and_precision)
    deltaTime = time.time() - start
    print('\r\tDone for {0:.0f}m {1:.0f}s'.format(deltaTime / 60, deltaTime % 60))
    for name, rcl, prc in zip(files_names, recalls, precisions):
        print('For {0} RECALL = {1:.2%}, PRECISION = {2:.2%}'.format(name, rcl, prc))
    print('Average RECALL = {0:.2%}, PRECISION = {1:.2%}'.format(np.average(recalls), np.average(precisions)))
    return recalls, precisions


