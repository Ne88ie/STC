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


@print_state_of_process(21196) # whole = int((t_features - window_width + 1)/step + 1)
def _delta_BICs(m1, m2, lam=3, use_diag=False):
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


def _points(local_min, d_bics, confidence_interval=30):
    """
    Точки в пределах доверительного интевала сливаем в одну точку, которая имеет минимальный delta_bic.
    :param local_min: массив позиций локальныз минимумов
    :param d_bics: массив зачений локальныз минимумов
    :param confidence_interval: интервал в пределах которого сливаем точки в одну
    :return: массив позиций точек
    """
    local_min, d_bics = map(np.ma.array, [local_min, d_bics])
    whole = local_min.count()
    points = []
    while d_bics.count() > 0:
        print('\r\tCounting masked points {0:.2%}'.format((whole - d_bics.count())/(whole - 1)), end='')
        least_unmasked_arg = d_bics.argmin()
        left, right = local_min[least_unmasked_arg] - confidence_interval, local_min[least_unmasked_arg] + confidence_interval
        points.append(local_min[least_unmasked_arg])
        local_min = np.ma.masked_inside(local_min, left, right)
        local_min[least_unmasked_arg] = np.ma.masked
        d_bics.mask = local_min.mask
    return np.array(sorted(points))


def _get_limit(d_bics, conf_int=1, procentage=0.25):
    d_bics = np.array([max(np.abs(d_bics[i-conf_int if i-conf_int > 0 else 0: i+conf_int+1] - bic)) for i, bic in enumerate(d_bics)])
    return d_bics.mean() + procentage * (d_bics.var()**0.5)


def _points2(local_min, d_bics):
    """
    Оставляем миниумы, имеющие отклонение больше определённого.
    :param local_min: массив позиций локальныз минимумов
    :param d_bics: массив зачений локальныз минимумов
    :return: массив позиций точек
    """
    whole = local_min.size
    conf_int = 1
    points = []

    number101, hist101 = np.histogram(d_bics, 101)
    i101_max1 = number101.argmax()

    number101 = np.ma.array(number101)
    number101[i101_max1] = np.ma.masked
    i101_max2 = number101.argmax()

    borders = [np.average(hist101[i101_max1:i101_max1+2]), np.average(hist101[i101_max2:i101_max2+2])]
    if np.argmax(borders) == 1:
        borders = borders[::-1]

    limit_up = _get_limit(d_bics[d_bics >= borders[0]], conf_int, 0.15)
    limit_medium = _get_limit(d_bics[np.all(np.vstack((d_bics < borders[0], d_bics > borders[1])), axis=0)], conf_int, 0.25)
    limit_down = _get_limit(d_bics[d_bics <= borders[1]], conf_int, 0.5)

    for i, l in enumerate(local_min):
        print('\r\tCounting limit points {0:.2%}'.format((i+1)/whole), end='')
        bic = d_bics[i]
        if bic >= borders[0] and np.any(np.abs(d_bics[i-conf_int if i-conf_int > 0 else 0: i+conf_int+1] - bic) > limit_up):
            points.append(l)
        elif bic < borders[0] and bic > borders[1] and np.any(np.abs(d_bics[i-conf_int if i-conf_int > 0 else 0: i+conf_int+1] - bic) > limit_medium):
            points.append(l)
        elif bic <= borders[1] and np.any(np.abs(d_bics[i-conf_int if i-conf_int > 0 else 0: i+conf_int+1] - bic) > limit_down):
            points.append(l)
    return np.array(points, dtype=int)


def find_points(features, window_width=120, step=15, lam=3, use_diag=False, confidence_interval=450):
    """
    По массиву признаков пробегаем окном шириной window_width с шагом step. Окно разбиваем на две половинки. Для точки -
    середины окна - рассчитываем _delta_BICs.
    :param features: список признаков, каждый из которых есть первые 20 коэффициентов MFCC (без С0)
    :param window_width: размер окна
    :param step: шаг
    :return: массив локальных максимумов
    """
    start = time.time()
    mid = window_width/2
    graphs = []

    delta_bics = np.array([_delta_BICs(features[i: i+mid], features[i+mid: i+window_width], lam, use_diag) # получаем масссив delta_bic-ов для всех точек,
                           for i in range(0, features.shape[0] - (window_width - 1), step)])
    graphs.append((np.arange(mid, mid+(delta_bics.size-1)*step+1, step), delta_bics, 'b.', 'delta_bics'))

    if window_width == 250: threshold = -1000000
    elif window_width == 200: threshold = -600000
    elif window_width == 160: threshold = -400000
    elif window_width == 150: threshold = -360000
    elif window_width == 120: threshold = -220000
    elif window_width == 100: threshold = -140000
    else: threshold = 0

    lether_zero = np.nonzero(delta_bics < threshold)[0] # берём отрицательные значения
    d_bics = np.array([0] + delta_bics[lether_zero].tolist() + [0]) # прибавляем к массиву нули слева и справа, чтобы потом было удобнее искать локальные максимумы
    local_min = np.array([lether_zero[i] for i, bic in enumerate(d_bics[1: -1]) if np.all(d_bics[i+1-1: i+1+2: 2] > bic)])
    d_bics = delta_bics[local_min]
    graphs.append((local_min * step + mid, d_bics, 'r.', 'local_min'))

    local_min = _points2(local_min, d_bics) # оставляем миниумы, имеющие отклонение больше определённого
    d_bics = delta_bics[local_min]

    local_min = _points(local_min, d_bics, confidence_interval/step) # точки в пределах доверительного интевала сливаем в одну точку, которая имеет минимальный delta_bic
    d_bics = delta_bics[local_min]
    local_min = (local_min * step + mid)
    graphs.append((local_min, d_bics, 'g.', 'confidence_int'))

    deltaTime = time.time() - start
    print('\r\tDone for {0:.0f}m {1:.0f}s'.format(deltaTime / 60, deltaTime % 60))
    return local_min, graphs


def _recal_and_precision(reference, getting):
    """
    Подсчитывае полноту и точность. Мы считаем, что точка смены диктора point определена,
    если есть пересечение с интервалом [point-3*step : point-3*step] включительно.
    :param reference: список идеальных точек
    :param getting: список полученных точек
    :param step: шаг, относительно него высчитываем доверительный интервал
    :return: два значения - полнота и точность
    """
    correct = 0
    for point in reference:
        interval_of_reference_points = np.array([point-i for i in range(1, 501)] + [point] + [point + i for i in range(1, 501)])
        if np.intersect1d(interval_of_reference_points, getting, assume_unique=True).size:
            correct += 1
    return correct/(reference.size), correct/(getting.size or 0.0000001)


def get_recals_and_precisions(reference_points, getting_points, files_names):
    """
    Подсчитываем полноту и точность для нескольких итераций.
    :param reference_points: список идеальных разметок
    :param getting_points: список полученных разметок
    :param files_names: имена файлов
    :return: списки recalls и precisions
    """
    recal_and_precision = [_recal_and_precision(ref, get) for ref, get in zip(reference_points, getting_points)]
    recalls, precisions = zip(*recal_and_precision)
    for name, rcl, prc in zip(files_names, recalls, precisions):
        print('For {0} RECALL = {1:.2%}, PRECISION = {2:.2%}'.format(name, rcl, prc))
    print('Average RECALL = {0:.2%}, PRECISION = {1:.2%}'.format(np.average(recalls), np.average(precisions)))
    return recalls, precisions


