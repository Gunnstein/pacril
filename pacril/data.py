# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import numpy as np
import scipy.stats as stats
from . import _load
from . import serialize


LOCOMOTIVES = {
    "1'C1't": {
        "a": {
            "xp": np.cumsum([0.0, 1.4, 2.6, 1.8, 1.8, 2.2, 1.5]),
            "p": np.array([10.875, 14.5, 14.5, 14.5, 10.875]),
            "speed": 75.0,
        },
    },
    "2'B-2": {
        "a": {
            "xp": np.cumsum([0.0, 1.0, 1.7, 1.8, 2.1, 2.2, 2.7, 1.2]),
            "p": np.array([7.5, 7.5, 10., 10., 5.25, 5.25]),
            "speed": 60.0,
        },
        "b": {
            "xp": np.cumsum([0.0, 1.0, 1.4, 2.0, 1.9, 2.2, 1.8, 1.3]),
            "p": np.array([5.25, 5.25, 7., 7., 5.25, 5.25]),
            "speed": 55.0,
        },
        "c": {
            "xp": np.cumsum([0.0, 1.3, 2.0, 2.0, 2.2, 2.7, 2.5, 1.7]),
            "p": np.array([8.25, 8.25, 11., 11., 8.25, 8.25]),
            "speed": 70.0,
        },
    },
    "1'C-3": {
        "a": {
            "xp": np.cumsum([0.0, 1.3, 2.6, 1.7, 1.7, 2.6, 1.5, 1.5, 1.5]),
            "p": np.array([8.25, 11., 11., 11., 8.25, 8.25, 8.25]),
            "speed": 60.0,
        },
        "b": {
            "xp": np.cumsum([0.0, 1.3, 1.8, 1.6, 1.8, 2.5, 1.4, 1.2, 1.4]),
            "p": np.array([5.25, 7., 7., 7., 5.25, 5.25, 5.25]),
            "speed": 60.0,
        },
        "c": {
            "xp": np.cumsum([0.0, 1.3, 2.6, 1.7, 2.1, 2.3, 1.5, 1.5, 1.5]),
            "p": np.array([7.5, 10., 10., 10., 7.5, 7.5, 7.5]),
            "speed": 60.0,
        }
    },
    "2'C-2'2'": {
        "a": {
            "xp": np.cumsum([0.0, 1.5, 2.3, 1.8, 1.8, 1.8,
                             2.6, 1.6, 1.3, 1.6, 1.2]),
            "p": np.array([11.25, 11.25, 15., 15., 15., 11.25, 11.25, 11.25,
                           11.25]),
            "speed": 90.0,
        },
        "b": {
            "xp": np.cumsum([0.0, 1.3, 2.1, 1.4, 1.7, 1.7,
                             2.0, 1.6, 1.2, 1.6, 1.1]),
            "p": np.array([9.0, 9.0, 12., 12., 12., 9.0, 9.0, 9.0, 9.0]),
            "speed": 65.0,
        },
    },
    "1'D-2'2'": {
        "a": {
            "xp": np.cumsum([0.0, 1.5, 2.5, 1.4, 1.4, 1.4,
                             2.6, 1.6, 1.3, 1.6, 1.2]),
            "p": np.array([9.0, 12., 12., 12., 12., 9.0, 9.0, 9.0, 9.0]),
            "speed": 45.0,
        },
        "b": {
            "xp": np.cumsum([0.0, 1.6, 2.6, 1.4, 1.4, 1.4,
                             2.7, 1.6, 1.3, 1.6, 1.2]),
            "p": np.array([10.75, 14., 14., 14., 14., 10.75, 10.75, 10.75,
                           10.75]),
            "speed": 45.0,
        },
        "c": {
            "xp": np.cumsum([0.0, 1.5, 2.5, 1.4, 1.4, 1.4,
                             2.8, 1.6, 1.3, 1.6, 1.2]),
            "p": np.array([11.25, 15., 15., 15., 15., 11.25, 11.25, 11.25,
                           11.25]),
            "speed": 40.0,
        },
        "d": {
            "xp": np.cumsum([0.0, 1.5, 2.5, 1.4, 1.4, 1.4,
                             3.0, 1.6, 1.3, 1.6, 1.2]),
            "p": np.array([11.25, 15., 15., 15., 15., 11.25, 11.25, 11.25,
                           11.25]),
            "speed": 45.0,
        },
    },
    "2'D-2'2'": {
        "a": {
            "xp": np.cumsum([0.0, 1.6, 2.1, 1.4, 2.0, 1.5, 1.5,
                             2.4, 1.6, 1.6, 1.6, 1.4]),
            "p": np.array([11.25, 11.25, 15., 15., 15., 15.,
                           11.25, 11.25, 11.25, 11.25]),
            "speed": 70.0,
        },
        "b": {
            "xp": np.cumsum([0.0, 1.4, 2.1, 1.4, 1.9, 1.5, 1.6,
                             2.2, 1.6, 1.3, 1.6, 1.2]),
            "p": np.array([9.0, 9.0, 12., 12., 12., 12.,
                           9.0, 9.0, 9.0, 9.0]),
            "speed": 70.0,
        },
    },
    "1'E-2'2'": {
        "a": {
            "xp": np.cumsum([0.0, 2.0, 2.6, 1.6, 1.6, 1.6, 1.6,
                             3.9, 1.8, 2.3, 1.8, 2.0]),
            "p": np.array([11.25, 15.0, 15., 15., 15., 15.,
                           11.25, 11.25, 11.25, 11.25]),
            "speed": 70.0,
        },
    },
    "B'B'": {
        "a": {
            "xp": np.cumsum([0., 2.0, 3.0, 2.8, 3.0, 2.0]),
            "p": np.array([15., 15., 15., 15.]),
            "speed": 70.0,
        },
        "b": {
            "xp": np.cumsum([0., 2.0, 3.2, 2.7, 3.2, 2.0]),
            "p": np.array([12., 12., 12., 12.]),
            "speed": 70.0,
        },
    },
    "Bo'Bo'": {
        "a": {
            "xp": np.cumsum([0., 2.0, 3.0, 4.4, 3.0, 2.0]),
            "p": np.array([16., 16., 16., 16.]),
            "speed": 105.0,
        },
        "b": {
            "xp": np.cumsum([0., 2.2, 3.2, 4.1, 3.2, 2.2]),
            "p": np.array([18., 18., 18., 18.]),
            "speed": 115.0,
        },
        "c": {
            "xp": np.cumsum([0., 2.4, 2.6, 8.4, 2.6, 2.4]),
            "p": np.array([21., 21., 21., 21.]),
            "speed": 200.0,
        },
        "d": {
            "xp": np.cumsum([0., 3.0, 2.4, 6.6, 2.4, 3.0]),
            "p": np.array([21., 21., 21., 21.]),
            "speed": 120.0,
        },
        "e": {
            "xp": np.cumsum([0., 2.6, 2.7, 5.0, 2.7, 2.6]),
            "p": np.array([20., 20., 20., 20.]),
            "speed": 140.0,
        },
        "f": {
            "xp": np.cumsum([0., 2.9, 2.6, 7.8, 2.6, 2.9]),
            "p": np.array([21., 21., 21., 21.]),
            "speed": 140.0,
        },
    },
    "Co'Co'": {
        "a": {
            "xp": np.cumsum([0., 2.2, 2.0, 2.0, 6.3, 2.0, 2.0, 2.2]),
            "p": np.array([17., 17., 17., 17., 17., 17.]),
            "speed": 143.0,
        },
        "b": {
            "xp": np.cumsum([0., 2.8, 1.8, 1.8, 4.8, 1.8, 1.8, 2.8]),
            "p": np.array([18., 18., 18., 18., 18., 18.]),
            "speed": 120.0,
        },
        "c": {
            "xp": np.cumsum([0., 2.6, 1.8, 2.1, 7.9, 2.1, 1.8, 2.6]),
            "p": np.array([20., 20., 20., 20., 20., 20.]),
            "speed": 160.0,
        },
        "d": {
            "xp": np.cumsum([0., 2.4, 1.8, 1.8, 11.0, 1.8, 1.8, 2.4]),
            "p": np.array([21., 21., 21., 21., 21., 21.]),
            "speed": 120.0,
        },
        "e": {
            "xp": np.cumsum([0., 2.1, 2.0, 2.1, 9.0, 2.1, 2.0, 2.1]),
            "p": np.array([21., 21., 21., 21., 21., 21.]),
            "speed": 120.0,
        },
        "f": {
            "xp": np.cumsum([0., 2.6, 1.8, 2.0, 7.9, 2.0, 1.8, 2.6]),
            "p": np.array([19., 19., 19., 19., 19., 19.]),
            "speed": 140.0,
        },
    },
}

WAGONS = {
    "passenger": {
        1: [{'N': 2, 'a': 2.3, 'b': 4.2, 'pmax': 9.0, 'pmin': 5.0},
            {'N': 2, 'a': 2.6, 'b': 4.2, 'pmax': 9.0, 'pmin': 5.0},
            {'N': 2, 'a': 2.8, 'b': 4.2, 'pmax': 9.0, 'pmin': 5.0},
            {'N': 4, 'a': 3.1, 'b': 11.3, 'c': 2.0, 'pmax': 9.0, 'pmin': 5.0},
            {'N': 4, 'a': 3.1, 'b': 11.8, 'c': 2.0, 'pmax': 9.0, 'pmin': 5.0},
            {'N': 4, 'a': 3.1, 'b': 11.3, 'c': 2.1, 'pmax': 9.0, 'pmin': 5.0},
            {'N': 4, 'a': 3.2, 'b': 11.2, 'c': 2.1, 'pmax': 9.0, 'pmin': 5.0},
            {'N': 4, 'a': 3.2, 'b': 12.0, 'c': 2.1, 'pmax': 9.0, 'pmin': 5.0}],
        2: [{'N': 4, 'a': 2.4, 'b': 13.4, 'c': 2.1, 'pmax': 11.0, 'pmin': 5.0},
            {'N': 4, 'a': 2.8, 'b': 13.2, 'c': 2.1, 'pmax': 11.0, 'pmin': 5.0},
            {'N': 4, 'a': 2.8, 'b': 14.4, 'c': 2.1, 'pmax': 11.0, 'pmin': 5.0},
            {'N': 4, 'a': 2.9, 'b': 12.5, 'c': 2.1, 'pmax': 11.0, 'pmin': 5.0},
            {'N': 4, 'a': 2.9, 'b': 13.5, 'c': 2.1, 'pmax': 11.0, 'pmin': 5.0},
            {'N': 4, 'a': 2.9, 'b': 13.7, 'c': 2.1, 'pmax': 11.0, 'pmin': 5.0},
            {'N': 4, 'a': 2.9, 'b': 13.8, 'c': 2.1, 'pmax': 11.0, 'pmin': 5.0},
            {'N': 4, 'a': 2.9, 'b': 13.8, 'c': 2.3, 'pmax': 11.0, 'pmin': 5.0},
            {'N': 4, 'a': 2.9, 'b': 11.6, 'c': 1.9, 'pmax': 11.0, 'pmin': 5.0},
            {'N': 4, 'a': 2.9, 'b': 13.1, 'c': 2.1, 'pmax': 11.0, 'pmin': 5.0},
            {'N': 4, 'a': 2.9, 'b': 13.4, 'c': 2.1, 'pmax': 11.0, 'pmin': 5.0},
            {'N': 4, 'a': 2.9, 'b': 13.6, 'c': 2.1, 'pmax': 11.0, 'pmin': 5.0},
            {'N': 4, 'a': 2.9, 'b': 14.1, 'c': 2.3, 'pmax': 11.0, 'pmin': 5.0},
            {'N': 4, 'a': 2.9, 'b': 14.4, 'c': 2.1, 'pmax': 11.0, 'pmin': 5.0},
            {'N': 4, 'a': 2.9, 'b': 14.4, 'c': 2.3, 'pmax': 11.0, 'pmin': 5.0},
            {'N': 4, 'a': 3.0, 'b': 13.6, 'c': 2.1, 'pmax': 11.0, 'pmin': 5.0},
            {'N': 4, 'a': 3.0, 'b': 14.4, 'c': 2.3, 'pmax': 11.0, 'pmin': 5.0},
            {'N': 4, 'a': 3.0, 'b': 14.1, 'c': 2.3, 'pmax': 11.0, 'pmin': 5.0},
            {'N': 4, 'a': 3.1, 'b': 14.0, 'c': 2.3, 'pmax': 11.0, 'pmin': 5.0},
            {'N': 4, 'a': 3.1, 'b': 14.2, 'c': 2.3, 'pmax': 11.0, 'pmin': 5.0},
            {'N': 4, 'a': 3.1, 'b': 14.1, 'c': 2.3, 'pmax': 11.0, 'pmin': 5.0},
            {'N': 4, 'a': 3.1, 'b': 14.4, 'c': 2.3, 'pmax': 11.0, 'pmin': 5.0}],
        3: [{'N': 4, 'a': 2.0, 'b': 16.0, 'c': 2.6, 'pmax': 12.0, 'pmin': 6.0},
            {'N': 4, 'a': 3.0, 'b': 14.4, 'c': 2.3, 'pmax': 12.0, 'pmin': 6.0},
            {'N': 4, 'a': 3.1, 'b': 13.4, 'c': 2.3, 'pmax': 12.0, 'pmin': 6.0},
            {'N': 4, 'a': 3.2, 'b': 11.8, 'c': 2.1, 'pmax': 12.0, 'pmin': 6.0},
            {'N': 4, 'a': 3.3, 'b': 14.4, 'c': 2.5, 'pmax': 12.0, 'pmin': 6.0},
            {'N': 4, 'a': 3.4, 'b': 14.4, 'c': 2.5, 'pmax': 12.0, 'pmin': 6.0},
            {'N': 4, 'a': 3.5, 'b': 14.5, 'c': 2.6, 'pmax': 12.0, 'pmin': 6.0},
            {'N': 4, 'a': 3.7, 'b': 16.0, 'c': 2.6, 'pmax': 12.0, 'pmin': 6.0},
            {'N': 4, 'a': 3.8, 'b': 15.0, 'c': 2.6, 'pmax': 12.0, 'pmin': 6.0},
            {'N': 4, 'a': 3.8, 'b': 15.5, 'c': 2.5, 'pmax': 12.0, 'pmin': 6.0},
            {'N': 4, 'a': 3.8, 'b': 15.5, 'c': 2.6, 'pmax': 12.0, 'pmin': 6.0},
            {'N': 4, 'a': 3.8, 'b': 16.0, 'c': 2.6, 'pmax': 12.0, 'pmin': 6.0},
            {'N': 4, 'a': 3.8, 'b': 16.0, 'c': 2.5, 'pmax': 12.0, 'pmin': 6.0},
            {'N': 4, 'a': 3.8, 'b': 13.2, 'c': 3.0, 'pmax': 12.0, 'pmin': 6.0},
            {'N': 4, 'a': 3.8, 'b': 13.2, 'c': 2.5, 'pmax': 12.0, 'pmin': 6.0},
            {'N': 4, 'a': 3.8, 'b': 13.4, 'c': 3.0, 'pmax': 12.0, 'pmin': 6.0}],
        4: [{'N': 4, 'a': 3.1, 'b': 13.4, 'c': 2.3, 'pmax': 13.0, 'pmin': 7.5},
            {'N': 4, 'a': 3.2, 'b': 11.8, 'c': 2.1, 'pmax': 13.0, 'pmin': 7.5},
            {'N': 4, 'a': 3.3, 'b': 14.4, 'c': 2.5, 'pmax': 13.0, 'pmin': 7.5},
            {'N': 4, 'a': 3.4, 'b': 14.4, 'c': 2.5, 'pmax': 13.0, 'pmin': 7.5},
            {'N': 4, 'a': 3.5, 'b': 14.5, 'c': 2.6, 'pmax': 13.0, 'pmin': 7.5},
            {'N': 4, 'a': 3.7, 'b': 16.0, 'c': 2.6, 'pmax': 13.0, 'pmin': 7.5},
            {'N': 4, 'a': 3.8, 'b': 15.0, 'c': 2.6, 'pmax': 13.0, 'pmin': 7.5},
            {'N': 4, 'a': 3.8, 'b': 15.5, 'c': 2.5, 'pmax': 13.0, 'pmin': 7.5},
            {'N': 4, 'a': 3.8, 'b': 15.5, 'c': 2.6, 'pmax': 13.0, 'pmin': 7.5},
            {'N': 4, 'a': 3.8, 'b': 16.0, 'c': 2.3, 'pmax': 13.0, 'pmin': 7.5},
            {'N': 4, 'a': 3.8, 'b': 16.0, 'c': 2.5, 'pmax': 13.0, 'pmin': 7.5},
            {'N': 4, 'a': 3.8, 'b': 16.0, 'c': 2.6, 'pmax': 13.0, 'pmin': 7.5},
            {'N': 4, 'a': 3.8, 'b': 13.2, 'c': 3.0, 'pmax': 13.0, 'pmin': 7.5},
            {'N': 4, 'a': 3.8, 'b': 13.2, 'c': 2.5, 'pmax': 13.0, 'pmin': 7.5},
            {'N': 4, 'a': 3.8, 'b': 13.4, 'c': 3.0, 'pmax': 13.0, 'pmin': 7.5},
            {'N': 4, 'a': 4.0, 'b': 16.0, 'c': 2.5, 'pmax': 13.0, 'pmin': 7.5},
            {'N': 4, 'a': 4.0, 'b': 17.4, 'c': 2.5, 'pmax': 13.0, 'pmin': 7.5},
            {'N': 4, 'a': 4.0, 'b': 18.2, 'c': 2.5, 'pmax': 13.0, 'pmin': 7.5}],
        5: [{'N': 4, 'a': 3.8, 'b': 16.0, 'c': 2.5, 'pmax': 14.0, 'pmin': 8.5},
            {'N': 4, 'a': 3.8, 'b': 16.0, 'c': 2.6, 'pmax': 14.0, 'pmin': 8.5},
            {'N': 4, 'a': 4.0, 'b': 16.0, 'c': 2.5, 'pmax': 14.0, 'pmin': 8.5},
            {'N': 4, 'a': 4.0, 'b': 19.1, 'c': 2.5, 'pmax': 14.0, 'pmin': 8.5},
            {'N': 4, 'a': 4.0, 'b': 17.4, 'c': 2.5, 'pmax': 14.0, 'pmin': 8.5},
            {'N': 4, 'a': 4.0, 'b': 18.2, 'c': 2.5, 'pmax': 14.0, 'pmin': 8.5}],
        6: [{'N': 4, 'a': 3.8, 'b': 16.0, 'c': 2.5, 'pmax': 14.0, 'pmin': 8.5},
            {'N': 4, 'a': 3.8, 'b': 16.0, 'c': 2.6, 'pmax': 14.0, 'pmin': 8.5},
            {'N': 4, 'a': 4.0, 'b': 16.0, 'c': 2.5, 'pmax': 14.0, 'pmin': 8.5},
            {'N': 4, 'a': 4.0, 'b': 19.1, 'c': 2.5, 'pmax': 14.0, 'pmin': 8.5},
            {'N': 4, 'a': 4.0, 'b': 17.4, 'c': 2.5, 'pmax': 14.0, 'pmin': 8.5},
            {'N': 4, 'a': 4.0, 'b': 18.2, 'c': 2.5, 'pmax': 14.0, 'pmin': 8.5}],
        },
    "freight": {
        1: [{'N': 2, 'a': 1.5, 'b': 2.9, 'pmax': 9.0, 'pmin': 2.3},
            {'N': 2, 'a': 1.6, 'b': 3.7, 'pmax': 9.0, 'pmin': 2.3},
            {'N': 2, 'a': 1.8, 'b': 2.8, 'pmax': 9.0, 'pmin': 2.3},
            {'N': 2, 'a': 1.9, 'b': 3.1, 'pmax': 9.0, 'pmin': 2.3},
            {'N': 2, 'a': 1.9, 'b': 3.0, 'pmax': 9.0, 'pmin': 2.3},
            {'N': 2, 'a': 1.9, 'b': 3.7, 'pmax': 9.0, 'pmin': 2.3},
            {'N': 2, 'a': 2.0, 'b': 3.7, 'pmax': 9.0, 'pmin': 2.3},
            {'N': 2, 'a': 2.0, 'b': 3.2, 'pmax': 9.0, 'pmin': 2.3},
            {'N': 2, 'a': 2.1, 'b': 3.8, 'pmax': 9.0, 'pmin': 2.3},
            {'N': 2, 'a': 2.2, 'b': 4.0, 'pmax': 9.0, 'pmin': 2.3},
            {'N': 2, 'a': 2.2, 'b': 3.8, 'pmax': 9.0, 'pmin': 2.3},
            {'N': 2, 'a': 2.3, 'b': 3.7, 'pmax': 9.0, 'pmin': 2.3},
            {'N': 2, 'a': 2.5, 'b': 4.0, 'pmax': 9.0, 'pmin': 2.3},
            {'N': 2, 'a': 2.5, 'b': 3.9, 'pmax': 9.0, 'pmin': 2.3}],
        2: [{'N': 2, 'a': 1.4, 'b': 2.8, 'pmax': 12.0, 'pmin': 3.0},
            {'N': 2, 'a': 1.5, 'b': 2.9, 'pmax': 12.0, 'pmin': 3.0},
            {'N': 2, 'a': 1.6, 'b': 3.7, 'pmax': 12.0, 'pmin': 3.0},
            {'N': 2, 'a': 1.8, 'b': 3.2, 'pmax': 12.0, 'pmin': 3.0},
            {'N': 2, 'a': 1.8, 'b': 2.0, 'pmax': 12.0, 'pmin': 3.0},
            {'N': 2, 'a': 1.9, 'b': 3.4, 'pmax': 12.0, 'pmin': 3.0},
            {'N': 2, 'a': 2.0, 'b': 3.7, 'pmax': 12.0, 'pmin': 3.0},
            {'N': 2, 'a': 2.1, 'b': 4.0, 'pmax': 12.0, 'pmin': 3.0},
            {'N': 2, 'a': 2.1, 'b': 3.7, 'pmax': 12.0, 'pmin': 3.0},
            {'N': 2, 'a': 2.3, 'b': 3.7, 'pmax': 12.0, 'pmin': 3.0},
            {'N': 2, 'a': 2.3, 'b': 4.0, 'pmax': 12.0, 'pmin': 3.0},
            {'N': 2, 'a': 2.4, 'b': 4.0, 'pmax': 12.0, 'pmin': 3.0},
            {'N': 2, 'a': 2.4, 'b': 4.4, 'pmax': 12.0, 'pmin': 3.0},
            {'N': 2, 'a': 2.5, 'b': 3.9, 'pmax': 12.0, 'pmin': 3.0},
            {'N': 2, 'a': 2.6, 'b': 4.5, 'pmax': 12.0, 'pmin': 3.0}],
        3: [{'N': 2, 'a': 1.6, 'b': 3.7, 'pmax': 15.0, 'pmin': 3.7},
            {'N': 2, 'a': 1.9, 'b': 3.2, 'pmax': 15.0, 'pmin': 3.7},
            {'N': 2, 'a': 1.9, 'b': 6.0, 'pmax': 15.0, 'pmin': 3.7},
            {'N': 2, 'a': 2.0, 'b': 3.2, 'pmax': 15.0, 'pmin': 3.7},
            {'N': 2, 'a': 2.1, 'b': 7.2, 'pmax': 15.0, 'pmin': 3.7},
            {'N': 2, 'a': 2.2, 'b': 5.3, 'pmax': 15.0, 'pmin': 3.7},
            {'N': 2, 'a': 2.4, 'b': 4.0, 'pmax': 15.0, 'pmin': 3.7},
            {'N': 2, 'a': 2.5, 'b': 5.9, 'pmax': 15.0, 'pmin': 3.7},
            {'N': 2, 'a': 2.7, 'b': 4.0, 'pmax': 15.0, 'pmin': 3.7},
            {'N': 2, 'a': 2.7, 'b': 6.0, 'pmax': 15.0, 'pmin': 3.7}],
        4: [{'N': 2, 'a': 1.6, 'b': 9.0, 'pmax': 20.0, 'pmin': 5.0},
            {'N': 2, 'a': 2.0, 'b': 7.5, 'pmax': 20.0, 'pmin': 5.0},
            {'N': 2, 'a': 2.3, 'b': 6.5, 'pmax': 20.0, 'pmin': 5.0},
            {'N': 2, 'a': 2.3, 'b': 9.0, 'pmax': 20.0, 'pmin': 5.0},
            {'N': 2, 'a': 2.4, 'b': 5.7, 'pmax': 20.0, 'pmin': 5.0},
            {'N': 2, 'a': 2.5, 'b': 9.0, 'pmax': 20.0, 'pmin': 5.0},
            {'N': 2, 'a': 2.6, 'b': 5.7, 'pmax': 20.0, 'pmin': 5.0},
            {'N': 2, 'a': 2.6, 'b': 9.0, 'pmax': 20.0, 'pmin': 5.0},
            {'N': 2, 'a': 2.7, 'b': 5.7, 'pmax': 20.0, 'pmin': 5.0},
            {'N': 2, 'a': 2.9, 'b': 8.0, 'pmax': 20.0, 'pmin': 5.0},
            {'N': 2, 'a': 3.0, 'b': 8.0, 'pmax': 20.0, 'pmin': 5.0},
            {'N': 2, 'a': 3.1, 'b': 8.0, 'pmax': 20.0, 'pmin': 5.0},
            {'N': 4, 'a': 2.5, 'b': 9.0, 'c': 1.8, 'pmax': 20.0, 'pmin': 5.0},
            {'N': 4, 'a': 2.5, 'b': 10.7, 'c': 1.8, 'pmax': 20.0, 'pmin': 5.0},
            {'N': 4, 'a': 2.5, 'b': 15.7, 'c': 1.8, 'pmax': 20.0, 'pmin': 5.0},
            {'N': 4, 'a': 3.2, 'b': 10.3, 'c': 1.8, 'pmax': 20.0, 'pmin': 5.0}],
        5: [{'N': 2, 'a': 2.0, 'b': 7.5, 'pmax': 22.5, 'pmin': 5.6},
            {'N': 2, 'a': 2.0, 'b': 8.5, 'pmax': 22.5, 'pmin': 5.6},
            {'N': 2, 'a': 2.4, 'b': 5.7, 'pmax': 22.5, 'pmin': 5.6},
            {'N': 2, 'a': 2.5, 'b': 9.0, 'pmax': 22.5, 'pmin': 5.6},
            {'N': 2, 'a': 2.6, 'b': 9.0, 'pmax': 22.5, 'pmin': 5.6},
            {'N': 2, 'a': 2.7, 'b': 9.3, 'pmax': 22.5, 'pmin': 5.6},
            {'N': 2, 'a': 2.9, 'b': 8.0, 'pmax': 22.5, 'pmin': 5.6},
            {'N': 2, 'a': 3.0, 'b': 8.0, 'pmax': 22.5, 'pmin': 5.6},
            {'N': 2, 'a': 3.1, 'b': 9.0, 'pmax': 22.5, 'pmin': 5.6},
            {'N': 2, 'a': 3.1, 'b': 8.0, 'pmax': 22.5, 'pmin': 5.6},
            {'N': 2, 'a': 3.3, 'b': 9.0, 'pmax': 22.5, 'pmin': 5.6},
            {'N': 2, 'a': 3.6, 'b': 10.0, 'pmax': 22.5, 'pmin': 5.6},
            {'N': 2, 'a': 3.8, 'b': 9.0, 'pmax': 22.5, 'pmin': 5.6},
            {'N': 2, 'a': 4.1, 'b': 9.0, 'pmax': 22.5, 'pmin': 5.6},
            {'N': 4, 'a': 2.5, 'b': 9.0, 'c': 1.8, 'pmax': 22.5, 'pmin': 5.6},
            {'N': 4, 'a': 2.5, 'b': 10.7, 'c': 1.8, 'pmax': 22.5, 'pmin': 5.6},
            {'N': 4, 'a': 2.5, 'b': 15.7, 'c': 1.8, 'pmax': 22.5, 'pmin': 5.6},
            {'N': 4, 'a': 3.2, 'b': 10.3, 'c': 1.8, 'pmax': 22.5, 'pmin': 5.6}],
        6: [{'N': 2, 'a': 2.0, 'b': 7.5, 'pmax': 22.5, 'pmin': 5.6},
            {'N': 2, 'a': 2.0, 'b': 8.5, 'pmax': 22.5, 'pmin': 5.6},
            {'N': 2, 'a': 2.3, 'b': 9.0, 'pmax': 22.5, 'pmin': 5.6},
            {'N': 2, 'a': 2.5, 'b': 9.0, 'pmax': 22.5, 'pmin': 5.6},
            {'N': 2, 'a': 2.6, 'b': 9.0, 'pmax': 22.5, 'pmin': 5.6},
            {'N': 2, 'a': 2.7, 'b': 9.3, 'pmax': 22.5, 'pmin': 5.6},
            {'N': 2, 'a': 2.9, 'b': 8.0, 'pmax': 22.5, 'pmin': 5.6},
            {'N': 2, 'a': 3.0, 'b': 8.0, 'pmax': 22.5, 'pmin': 5.6},
            {'N': 2, 'a': 3.1, 'b': 9.0, 'pmax': 22.5, 'pmin': 5.6},
            {'N': 2, 'a': 3.1, 'b': 8.0, 'pmax': 22.5, 'pmin': 5.6},
            {'N': 2, 'a': 3.3, 'b': 9.0, 'pmax': 22.5, 'pmin': 5.6},
            {'N': 2, 'a': 3.6, 'b': 10.0, 'pmax': 22.5, 'pmin': 5.6},
            {'N': 2, 'a': 3.8, 'b': 9.0, 'pmax': 22.5, 'pmin': 5.6},
            {'N': 2, 'a': 4.1, 'b': 9.0, 'pmax': 22.5, 'pmin': 5.6},
            {'N': 4, 'a': 2.5, 'b': 8.5, 'c': 1.8, 'pmax': 22.5, 'pmin': 5.6},
            {'N': 4, 'a': 2.5, 'b': 9.0, 'c': 1.8, 'pmax': 22.5, 'pmin': 5.6},
            {'N': 4, 'a': 2.5, 'b': 15.7, 'c': 1.8, 'pmax': 22.5, 'pmin': 5.6},
            {'N': 4, 'a': 2.5, 'b': 9.9, 'c': 1.8, 'pmax': 22.5, 'pmin': 5.6},
            {'N': 4, 'a': 2.5, 'b': 10.7, 'c': 1.8, 'pmax': 22.5, 'pmin': 5.6},
            {'N': 4, 'a': 3.2, 'b': 10.3, 'c': 1.8, 'pmax': 22.5, 'pmin': 5.6},
            {'N': 6, 'a': 2.5, 'b': 14.9, 'c': 1.8, 'pmax': 22.5, 'pmin': 5.6},
            {'N': 6, 'a': 2.8, 'b': 14.2, 'c': 1.8, 'pmax': 22.5, 'pmin': 5.6},
            {'N': 6, 'a': 2.8, 'b': 14.4, 'c': 1.8, 'pmax': 22.5, 'pmin': 5.6}]
        }
}


INFLUENCELINES = {
    "Hell": {
        "fx": 10.,
        "crossgirder": np.array([
        -0.03, -0.02, -0.02, -0.01, -0.05, -0.01, -0.04, -0.04, -0.04, -0.04,
        -0.05,  0.00, -0.04, -0.01, -0.06,  0.01, -0.06, -0.02, -0.04, -0.01,
        -0.00,  0.01,  0.01,  0.03,  0.05,  0.04,  0.01,  0.02, -0.00,  0.00,
         0.00,  0.02,  0.02,  0.03,  0.04,  0.06,  0.03,  0.01,  0.01, -0.01,
        -0.01, -0.02, -0.01, -0.02, -0.02, -0.01, -0.00, -0.01, -0.03, -0.02,
        -0.03, -0.06, -0.06, -0.07, -0.04, -0.05, -0.03, -0.01, -0.01, -0.01,
        -0.02, -0.02, -0.02, -0.04, -0.02, -0.04, -0.00, -0.02,  0.03,  0.02,
         0.03,  0.02,  0.02, -0.02,  0.00, -0.00,  0.04,  0.02,  0.07,  0.06,
         0.11,  0.10,  0.12,  0.12,  0.15,  0.13,  0.18,  0.20,  0.25,  0.25,
         0.33,  0.36,  0.38,  0.42,  0.48,  0.51,  0.58,  0.62,  0.69,  0.74,
         0.81,  0.87,  0.95,  0.99,  1.02,  1.07,  1.15,  1.23,  1.28,  1.34,
         1.43,  1.47,  1.55,  1.57,  1.63,  1.64,  1.69,  1.72,  1.74,  1.76,
         1.81,  1.84,  1.88,  1.88,  1.91,  1.95,  1.90,  1.87,  1.84,  1.82,
         1.75,  1.75,  1.71,  1.68,  1.66,  1.66,  1.61,  1.59,  1.53,  1.53,
         1.43,  1.44,  1.35,  1.34,  1.29,  1.27,  1.22,  1.19,  1.11,  1.06,
         0.97,  0.95,  0.86,  0.83,  0.76,  0.73,  0.65,  0.62,  0.56,  0.51,
         0.42,  0.40,  0.31,  0.27,  0.21,  0.18,  0.14,  0.11,  0.07,  0.05,
         0.02, -0.00, -0.04, -0.03, -0.07, -0.07, -0.09, -0.10, -0.10, -0.10,
        -0.08, -0.08, -0.10, -0.09, -0.11, -0.08, -0.09, -0.10, -0.12, -0.09,
        -0.09, -0.09, -0.07, -0.08, -0.07, -0.07, -0.09, -0.07, -0.10, -0.07,
        -0.05, -0.05, -0.02, -0.04,  0.01, -0.02, -0.01, -0.02, -0.02, -0.05,
         0.01, -0.02, -0.00, -0.00,  0.01, -0.01, -0.01, -0.01, -0.01, -0.04,
        -0.01, -0.03, -0.02, -0.02,  0.00, -0.00,  0.02, -0.02, -0.02, -0.02,
        -0.03, -0.03, -0.03, -0.04, -0.04, -0.08, -0.04, -0.08, -0.07, -0.09,
        -0.09, -0.11, -0.11, -0.13, -0.10, -0.10, -0.09, -0.11, -0.08, -0.09,
        -0.07, -0.07, -0.05, -0.07, -0.09, -0.08, -0.06, -0.08, -0.03, -0.06,
        -0.02, -0.05, -0.03, -0.05, -0.04, -0.00, -0.02, -0.02, -0.01,  0.01,
         0.00,  0.02,  0.02,  0.05,  0.02,  0.04,  0.03,  0.05,  0.03,  0.05,
         0.04,  0.04,  0.00,  0.02,  0.01,  0.02, -0.02,  0.02, -0.01,  0.02,
        -0.01,  0.01,  0.01, -0.00, -0.01, -0.01, -0.03, -0.03, -0.05, -0.03,
        -0.05, -0.02, -0.05, -0.03, -0.04, -0.03, -0.07, -0.07, -0.11, -0.09,
        -0.12, -0.09, -0.14, -0.11, -0.09, -0.08, -0.08, -0.07, -0.08, -0.06,
        -0.10, -0.06, -0.11, -0.08, -0.10, -0.09, -0.10, -0.11, -0.10, -0.10,
        -0.10, -0.10, -0.09, -0.08, -0.08, -0.07, -0.05, -0.07, -0.03, -0.05,
        -0.03, -0.05, -0.02, -0.00,  0.01,  0.02,  0.02,  0.04,  0.05,  0.03,
         0.05,  0.04,  0.04,  0.03,  0.05,  0.07,  0.06,  0.09,  0.08,  0.10,
         0.11,  0.12,  0.12,  0.11,  0.11,  0.11,  0.13,  0.11,  0.12,  0.12,
         0.14,  0.11,  0.14,  0.10,  0.14,  0.10,  0.14,  0.10,  0.12,  0.09,
         0.12,  0.08,  0.13,  0.07,  0.13,  0.07,  0.10,  0.07,  0.10,  0.09,
         0.10,  0.09]),
        "stringer": np.array([
        -0.00,  0.01, -0.00, -0.01,  0.02, -0.02, -0.00,  0.01,  0.01, -0.00,
        -0.00, -0.01, -0.02, -0.03, -0.05, -0.07, -0.05, -0.06, -0.04, -0.05,
        -0.04, -0.03, -0.02,  0.02, -0.01,  0.06,  0.08,  0.13,  0.18,  0.26,
         0.35,  0.43,  0.50,  0.60,  0.68,  0.81,  0.88,  1.04,  1.14,  1.25,
         1.33,  1.48,  1.53,  1.64,  1.71,  1.82,  1.84,  1.91,  1.95,  1.97,
         1.97,  1.94,  1.90,  1.85,  1.75,  1.66,  1.55,  1.44,  1.30,  1.18,
         1.08,  0.93,  0.81,  0.70,  0.60,  0.48,  0.39,  0.30,  0.19,  0.10,
        -0.02, -0.09, -0.18, -0.22, -0.28, -0.30, -0.35, -0.37, -0.40, -0.41,
        -0.45, -0.42, -0.44, -0.38, -0.43, -0.36, -0.35, -0.34, -0.31, -0.29,
        -0.30, -0.30, -0.29, -0.30, -0.27, -0.25, -0.26, -0.24, -0.24, -0.25,
        -0.26, -0.26, -0.25, -0.27, -0.24, -0.23, -0.22, -0.21, -0.19, -0.20,
        -0.17, -0.14, -0.12, -0.08, -0.06, -0.03, -0.02,  0.01,  0.02,  0.04,
         0.07,  0.08,  0.08,  0.13,  0.11,  0.13,  0.11,  0.13,  0.09,  0.12,
         0.11,  0.10,  0.10,  0.09,  0.11,  0.08,  0.08,  0.04,  0.05,  0.03,
         0.02,  0.02,  0.01,  0.00,  0.02, -0.05, -0.00, -0.11, -0.07, -0.11,
        -0.10, -0.12, -0.10, -0.09, -0.07, -0.08, -0.09, -0.07, -0.06, -0.04,
        -0.04, -0.03,  0.01,  0.02,  0.02,  0.03,  0.01,  0.03,  0.01,  0.03,
         0.01,  0.04,  0.04,  0.02,  0.03, -0.01, -0.01, -0.02, -0.04, -0.05,
        -0.05, -0.06, -0.08, -0.10, -0.09, -0.13, -0.10, -0.12, -0.09, -0.10,
        -0.06, -0.09, -0.06, -0.08, -0.07, -0.06, -0.06, -0.05, -0.05, -0.03,
        -0.03, -0.04, -0.00, -0.02,  0.00, -0.01,  0.02,  0.04,  0.05,  0.11,
         0.09,  0.13,  0.11,  0.13,  0.11,  0.11,  0.10,  0.11,  0.10,  0.10,
         0.07,  0.07,  0.05,  0.03,  0.00, -0.01, -0.04, -0.03, -0.02, -0.05,
        -0.04, -0.06, -0.04, -0.06, -0.06, -0.06, -0.05, -0.03, -0.03, -0.02,
        -0.01, -0.01, -0.01, -0.02,  0.00, -0.02, -0.00,  0.03,  0.02,  0.05,
         0.03,  0.05,  0.03,  0.06,  0.04,  0.08,  0.06,  0.10,  0.09,  0.10,
         0.10,  0.09,  0.10,  0.10,  0.06,  0.08,  0.07,  0.07,  0.05,  0.07,
         0.03,  0.03,  0.01,  0.02,  0.01,  0.03,  0.05,  0.06,  0.07,  0.07,
         0.08,  0.10,  0.11,  0.10,  0.13,  0.14,  0.14,  0.16,  0.16,  0.13,
         0.12,  0.13,  0.12,  0.11,  0.12,  0.12,  0.11,  0.12,  0.11,  0.12,
         0.11,  0.11,  0.11,  0.13,  0.10,  0.13,  0.10,  0.12,  0.05,  0.06,
         0.03,  0.03, -0.00,  0.02, -0.01, -0.01, -0.03, -0.03, -0.04, -0.06,
        -0.04, -0.05, -0.01, -0.02, -0.00,  0.00,  0.02,  0.02,  0.02,  0.02,
         0.02,  0.03,  0.04,  0.04,  0.04,  0.03,  0.05,  0.03,  0.02,  0.00,
        -0.01,  0.00,  0.00,  0.00, -0.00,  0.01, -0.00, -0.00, -0.01, -0.03,
        -0.01, -0.01, -0.01, -0.02, -0.00, -0.01, -0.03, -0.04, -0.06, -0.06,
        -0.05, -0.05, -0.06, -0.05, -0.05, -0.04, -0.05, -0.04, -0.05, -0.03,
        -0.03, -0.02,  0.00, -0.00,  0.02, -0.01,  0.01, -0.02, -0.00, -0.02,
         0.01, -0.01, -0.01, -0.02, -0.02, -0.04, -0.04, -0.06, -0.06, -0.07])
    }
}


class NorwegianLocomotive(_load.Locomotive):
    """Define a Norwegian locomotive by its litra and sublitra.

    For more information, see

    G. Frøseth, A. Rønnquist. Evolution of load conditions in the Norwegian
        railway network and imprecision of historic railway loading data.
        Structure and Infrastructure. 2018


    Arguments
    ---------
    litra, sublitra: str
        The litra (e.g "B'B'") and sublitra (e.g "a") for the locomotives
        defined in the data module
    """
    def __init__(self, litra, sublitra):
        self.litra = litra
        self.sublitra = sublitra
        loc = LOCOMOTIVES[litra][sublitra]
        self.speed = loc["speed"]
        super(NorwegianLocomotive, self).__init__(loc['xp'], loc['p'])

    def todict(self):
        d = super(NorwegianLocomotive, self).todict()
        d["litra"] = self.litra
        d["sublitra"] = self.sublitra
        d["speed"] = self.speed
        return d

    def __str__(self):
        return "{0}({1},{2})".format(type(self).__name__, self.litra,
                                     self.sublitra)


class NorwegianRollingStock(_load.RollingStock):
    """Rolling stock for different periods and train types in Norway

    The rolling stock for different periods in the Norwegian railway network
    is defined for the local suburban (ls), passenger trains (p) and freight
    trains (f). There are a total of six different periods
        =======================    ====================================
        | period |    Years   |    | traintype |    Train             |
        |--------+------------|    |-----------+----------------------|
        |   1    |     --1900 |    |     p     | Passenger train      |
        |   2    | 1900--1930 |    |     f     | Freight train        |
        |   3    | 1930--1930 |    ====================================
        |   4    | 1960--1985 |
        |   5    | 1985--2000 |
        |   6    | 2000--     |
        =======================

    For more information, see

    G. Frøseth, A. Rønnquist. Evolution of load conditions in the Norwegian
        railway network and imprecision of historic railway loading data.
        Structure and Infrastructure. 2018

    Arguments
    ---------
    period : int
        The period to get the rolling stock from.
    traintype : str
        The train type, see the table above.
    """
    def __init__(self, period, traintype):
        self.period = period
        self.traintype = traintype.lower()
        locs, wagons = [], []
        self.Nmin, self.Nmax = None, None
        self.pmin, self.pmax = None, None
        if self.traintype == "p":
            if self.period == 1:
                self.maxspeed = 70
                self.Nmin, self.Nmax = 1, 20
                locs = [NorwegianLocomotive("2'B-2", sublitra)
                        for sublitra in ["a", "b", "c"]]
            elif self.period == 2:
                self.maxspeed = 90
                self.Nmin, self.Nmax = 1, 20
                locs = ([NorwegianLocomotive("2'B-2", sublitra)
                        for sublitra in ["a", "b", "c"]]
                        + [NorwegianLocomotive("2'C-2'2'", sublitra)
                           for sublitra in ["a", "b"]]
                        + [NorwegianLocomotive("2'D-2'2'", sublitra)
                           for sublitra in ["a", "b"]])
            elif self.period == 3:
                self.Nmin, self.Nmax = 2, 20
                self.maxspeed = 90
                locs = ([NorwegianLocomotive("2'C-2'2'", sublitra)
                         for sublitra in ["a", "b"]]
                        + [NorwegianLocomotive("2'D-2'2'", sublitra)
                           for sublitra in ["a", "b"]]
                        + [NorwegianLocomotive("B'B'", sublitra)
                           for sublitra in ["a", "b"]])
            elif self.period == 4:
                self.Nmin, self.Nmax = 3, 20
                self.maxspeed = 120
                locs = ([NorwegianLocomotive("B'B'", sublitra)
                         for sublitra in ["a", "b"]]
                        + [NorwegianLocomotive("Bo'Bo'", sublitra)
                           for sublitra in ["a", "b"]]
                        + [NorwegianLocomotive("Co'Co'", sublitra)
                           for sublitra in ["a", "b"]])
            elif self.period == 5 or self.period == 6:
                self.maxspeed = 160
                self.Nmin, self.Nmax = 5, 20
                sublitras = ["a", "b", "c", "d", "e", "f"]
                if self.period == 6:
                    sublitras = sublitras[2:]
                locs = ([NorwegianLocomotive("Bo'Bo'", sublitra)
                         for sublitra in sublitras]
                        + [NorwegianLocomotive("Co'Co'", sublitra)
                           for sublitra in sublitras])
        elif self.traintype == "f":
            self.Nmin, self.Nmax = 10, 50
            if self.period == 1:
                self.maxspeed = 50
                locs = [NorwegianLocomotive("1'C-3", sublitra)
                        for sublitra in ["a", "b", "c"]]
            elif self.period == 2:
                self.maxspeed = 65
                locs = ([NorwegianLocomotive("1'C-3", sublitra)
                        for sublitra in ["a", "b", "c"]]
                        + [NorwegianLocomotive("1'D-2'2'", sublitra)
                        for sublitra in ["a", "b", "c", "d"]])
            elif self.period == 3:
                self.maxspeed = 65
                locs = ([NorwegianLocomotive("1'D-2'2'", sublitra)
                         for sublitra in ["a", "b", "c", "d"]]
                        + [NorwegianLocomotive("1'E-2'2'", "a")]
                        + [NorwegianLocomotive("B'B'", sublitra)
                           for sublitra in ["a", "b"]])
            elif self.period == 4:
                self.maxspeed = 80
                locs = ([NorwegianLocomotive("B'B'", sublitra)
                         for sublitra in ["a", "b"]]
                        + [NorwegianLocomotive("Bo'Bo'", sublitra)
                           for sublitra in ["a", "b"]]
                        + [NorwegianLocomotive("Co'Co'", sublitra)
                           for sublitra in ["a", "b"]])
                self.pmax = 18.0
            elif self.period == 5 or self.period == 6:
                self.maxspeed = 80
                sublitras = ["a", "b", "c", "d", "e", "f"]
                if self.period == 6:
                    self.maxspeed = 90
                    sublitras = sublitras[2:]
                locs = [NorwegianLocomotive(litra, sublitra)
                        for litra in ["Bo'Bo'", "Co'Co'"]
                        for sublitra in sublitras]
        else:
            raise ValueError("Invalid train type")
        wagons = self._get_wagons(self.period, self.traintype)
        super(NorwegianRollingStock, self).__init__(locs, wagons)

    def get_neighbor_train(self, train, fixed_length_trains=False,
                           Nwag_min=None, Nwag_max=None):
        Nmin = Nwag_min or self.Nmin
        Nmax = Nwag_max or self.Nmax
        return super(NorwegianRollingStock, self).get_neighbor_train(
            train, fixed_length_trains=fixed_length_trains, Nwag_min=Nmin,
            Nwag_max=Nmax)

    def _get_wagons(self, period, traintype, loadlevels=5):
        if traintype == 'p':
            ttp = 'passenger'
        elif traintype == 'f':
            ttp = 'freight'
        wagons = []
        for w in WAGONS[ttp][period]:
            a, b, N = w["a"], w["b"], w["N"]
            pmin = self.pmin or w["pmin"]
            pmax = self.pmax or w["pmax"]
            ps = np.linspace(pmin, pmax, loadlevels)
            for psi in ps:
                p = np.array([psi]*N)
                pem = np.array([pmin]*N)
                if N == 2:
                    W = _load.TwoAxleWagon(p, a, b, pem)
                elif N == 4:
                    c = w["c"]
                    W = _load.BogieWagon(p, a, b, c, pem)
                elif N == 6:
                    c = w["c"]
                    W = _load.JacobsWagon(p, a, b, c, pem)
                wagons.append(W)
        return wagons

    def _get_geometry(self, x1, x2):
        """Return a geometry set from bounds 'x1' and 'x2'

        Get thirteen points evenly distributed over the domain defined
        by the interval vectors 'x1' and 'x2'.

        Arguments
        ---------
        x1,x2 : list
            The lower and upper bounds for the two variables x1 and x2.

        Returns
        -------
        ndarray
            The geometry points (x1i, x2i) distributed over the domain
            of the interval variables.
        """
        x, y = np.linspace(x1[0], x1[1], 5), np.linspace(x2[0], x2[1], 5)
        return np.array([[x[0], y[0]], [x[4], y[0]], [x[4], y[4]],
                         [x[0], y[4]], [x[2], y[2]], [x[2], y[0]],
                         [x[4], y[2]], [x[2], y[4]], [x[0], y[2]],
                         [x[1], y[1]], [x[3], y[1]], [x[3], y[3]],
                         [x[1], y[3]]])

    def todict(self):
        d = super(NorwegianRollingStock, self).todict()
        d["period"] = self.period
        d["traintype"] = self.traintype
        return d


class PacrilJSONDecoder(serialize.PacrilJSONDecoder):
    def object_hook(self, obj):
        if "pacrilcls" not in obj:
            return obj
        pacrilcls = obj["pacrilcls"]
        if pacrilcls == "NorwegianLocomotive":
            return NorwegianLocomotive(obj["litra"], obj["sublitra"])
        elif pacrilcls == "NorwegianRollingStock":
            return NorwegianRollingStock(obj["period"], obj["traintype"])
        else:
            return super(PacrilJSONDecoder, self).object_hook(obj)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    rs = NorwegianRollingStock(6, 'f')
    print(rs)
    plt.figure(dpi=144)
    for n in range(3):
        train = rs.get_train(np.random.randint(10, 50))
        l = INFLUENCELINES['Hell']['stringer']
        z = train.apply(l)
        plt.plot(z, label="Train {0}".format(n+1))
    plt.legend()
    plt.show(block=True)
