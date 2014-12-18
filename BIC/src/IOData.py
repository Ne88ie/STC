# coding=utf-8
__author__ = 'annie'

from struct import pack, unpack
import numpy as np
import os


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
