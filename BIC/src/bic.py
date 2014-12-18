# coding=utf-8
from __future__ import print_function, division
from math import ceil
import numpy as np
import time

__author__ = 'annie'

'''
dim = 20
t_features = 318045
'''

def print_times_of_calls(msg, whole):
    def my_decorator(func):
        def wrapped(*args, **kwargs):
            wrapped.time += 1
            print('\r\t{0} {1:.2%}'.format(msg, wrapped.time/whole), end='')
            if wrapped.time == whole:
                wrapped.time = 0
            return func(*args, **kwargs)
        wrapped.time = 0
        return wrapped
    return my_decorator


@print_times_of_calls('counting delta BIC', 31785)
def delts_BIC(m1, m2, lam=3.05):
    n1 = m1.shape[0]
    n2 = m2.shape[0]
    n12 = n1 + n2
    dim = m1.shape[1]
    beta = dim * (dim + 1)/2
    return 0.5 * (n1 * np.multiply(*np.linalg.slogdet(np.cov(m1))) +
                  n2 * np.multiply(*np.linalg.slogdet(np.cov(m2))) -
                  n12 * np.multiply(*np.linalg.slogdet(np.cov(np.vstack((m1, m2))))) +
                  lam * beta * np.log(n12))


def run(features, window_width=200, step=10):
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
    d_bics = np.array([0] + [delts_BIC(features[i: i+mid], features[i+mid: i+window_width])
                       for i in range(0, features.shape[0] - (window_width - 1), step)] + [0])
    local_max = np.array([i-1 for i in np.nonzero(d_bics > 0)[0] if np.all(d_bics[i-1: i+2: 2] < d_bics[i])])
    deltaTime = time.time() - start
    print('\r\tDone for {0:.0f}m {1:.0f}s'.format(deltaTime / 60, deltaTime % 60))
    return local_max[1:-1]


