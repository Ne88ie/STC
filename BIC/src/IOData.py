# coding=utf-8
__author__ = 'annie'

from struct import pack, unpack
import matplotlib.pyplot as plt
import numpy as np

class Graph:
    def __init__(self, number):
        self.plots = [[] for i in range(number)]
    def add_plot(self, num_graphic, x, y, atr, label):
        self.plots[num_graphic].append((x, y, atr, label))
    def show(self, save_to, show_ref=True):
        plt.figure(1)
        for i, sample in enumerate(self.plots):
            plt.subplot(221 + i)
            plt.title('Sample {0}'.format(i+1))
            for curve in sample:
                if curve:
                    plt.plot(*curve[: -1])
            if show_ref:
                min_points = np.min(sample[2][1])
                for p in sample[-1][0]:
                    plt.plot([p]*2,
                             [-min_points, min_points],
                             'm-')
        plt.savefig(save_to, format='png')
        plt.show()


def _read_matrix(file, row, col):
    return np.array(unpack('<{0}f'.format(row*col), file.read(4*row*col)), dtype=np.float64).reshape(row, col)


def get_features(path):
    with open(path, 'rb') as f:
        dim = unpack('<i', f.read(4))[0]
        t_features = unpack('<i', f.read(4))[0]
        return _read_matrix(f, t_features, dim)


def get_points(path):
    with open(path, 'rb') as f:
        k_points = unpack('<i', f.read(4))[0]
        return np.array(unpack('<{0}i'.format(k_points), f.read(4*k_points)), dtype=np.int)


def save_points(path, points):
    with open(path, 'wb') as f:
        f.write(pack('<i', points.size))
        f.write(pack('<{0}i'.format(points.size), *points.ravel()))
    print('\tsaved points')


class Logger:
    def __init__(self, path, start=1, rewrite=False):
        self.path = path
        self.number = start
        if rewrite:
            with open(path, 'w') as f:
                f.write('sample;window;step;lam;use_diag;rcl1;rcl2;rcl3;rcl4;rclAve;prc1;prc2;prc3;prc4;prcAve;conf\n')

    def write(self, window_width, step, lam, use_diag, recalls, precisions, confidence_interval):
        with open(self.path, 'a') as f:
            f.write(';'.join(map(str, [self.number, window_width, step, lam, use_diag] +
                                 map(lambda x: round(100 * x, 2),
                                     list(recalls) + [np.average(recalls)] +
                                     list(precisions) + [np.average(precisions)]) +
                                 [confidence_interval])) + '\n')
            self.number += 1




