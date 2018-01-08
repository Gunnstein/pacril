# -*- coding: utf-8 -*-
import numpy as np


__all__ = ['get_coordinate_vector']


def get_coordinate_vector(y, fx=10.):
    return np.arange(np.asfarray(y).size, dtype=float) / float(fx)
