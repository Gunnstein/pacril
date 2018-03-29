# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import numpy as np
import scipy.stats as stats
import pacril._load as _load
import pacril.serialize as _serialize
from pacril.serialize import PacrilJSONEncoder


LOCOMOTIVES = {
    "1'C1't": {
        "a" : {
            "xp": np.cumsum([0.0, 1.4, 2.6, 1.8, 1.8, 2.2, 1.5]),
            "p": np.array([10.875, 14.5, 14.5, 14.5, 10.875]),
        },
    },
    "2'B-2": {
        "a": {
            "xp": np.cumsum([0.0, 1.0, 1.7, 1.8, 2.1, 2.2, 2.7, 1.2]),
            "p": np.array([7.5, 7.5, 10., 10., 10., 10.]),
        },
        "b": {
            "xp": np.cumsum([0.0, 1.0, 1.4, 2.0, 1.9, 2.2, 1.8, 1.3]),
            "p": np.array([5.25, 5.25, 7., 7., 7., 7.]),
        },
        "c": {
            "xp": np.cumsum([0.0, 1.3, 2.0, 2.0, 2.2, 2.7, 2.5, 1.7]),
            "p": np.array([8.25, 11., 11., 11., 11.]),
        },
    },
    "1'C-3": {
        "a": {
            "xp": np.cumsum([0.0, 1.3, 2.6, 1.7, 1.7, 2.6, 1.5, 1.5, 1.5]),
            "p": np.array([8.25, 11., 11., 11., 11., 11., 11.]),
        },
        "b": {
            "xp": np.cumsum([0.0, 1.3, 1.8, 1.6, 1.8, 2.5, 1.4, 1.2, 1.4]),
            "p": np.array([5.25, 7., 7., 7., 7., 7., 7.]),
        },
        "c": {
            "xp": np.cumsum([0.0, 1.3, 2.6, 1.7, 2.1, 2.3, 1.5, 1.5, 1.5]),
            "p": np.array([7.5, 10., 10., 10., 10., 10., 10.]),
        }
    },
    "2'C-2'2'": {
        "a": {
            "xp": np.cumsum([0.0, 1.5, 2.3, 1.8, 1.8, 1.8,
                             2.6, 1.6, 1.3, 1.6, 1.2]),
            "p": np.array([11.25, 11.25, 15., 15., 15., 15., 15., 15., 15.]),
        },
        "b": {
            "xp": np.cumsum([0.0, 1.3, 2.1, 1.4, 1.7, 1.7,
                             2.0, 1.6, 1.2, 1.6, 1.1]),
            "p": np.array([9.0, 9.0, 12., 12., 12., 12., 12., 12., 12.]),
        },
    },
    "1'D-2'2'": {
        "a": {
            "xp": np.cumsum([0.0, 1.5, 2.5, 1.4, 1.4, 1.4,
                             2.6, 1.6, 1.3, 1.6, 1.2]),
            "p": np.array([9.0, 12., 12., 12., 12., 12., 12., 12., 12.]),
        },
        "b": {
            "xp": np.cumsum([0.0, 1.6, 2.6, 1.4, 1.4, 1.4,
                             2.7, 1.6, 1.3, 1.6, 1.2]),
            "p": np.array([10.75, 14., 14., 14., 14., 14., 14., 14., 14.]),
        },
        "c": {
            "xp": np.cumsum([0.0, 1.5, 2.5, 1.4, 1.4, 1.4,
                             2.8, 1.6, 1.3, 1.6, 1.2]),
            "p": np.array([11.25, 15., 15., 15., 15., 15., 15., 15., 15.]),
        },
        "d": {
            "xp": np.cumsum([0.0, 1.5, 2.5, 1.4, 1.4, 1.4,
                             3.0, 1.6, 1.3, 1.6, 1.2]),
            "p": np.array([11.25, 15., 15., 15., 15., 15., 15., 15., 15.]),
        },
    },
    "2'D-2'2'": {
        "a": {
            "xp": np.cumsum([0.0, 1.6, 2.1, 1.4, 2.0, 1.5, 1.5,
                             2.4, 1.6, 1.6, 1.6, 1.4]),
            "p": np.array([11.25, 11.25, 15., 15., 15., 15.,
                           15., 15., 15., 15.]),
        },
        "b": {
            "xp": np.cumsum([0.0, 1.4, 2.1, 1.4, 1.9, 1.5, 1.6,
                             2.2, 1.6, 1.3, 1.6, 1.2]),
            "p": np.array([9.0, 9.0, 12., 12., 12., 12.,
                           12., 12., 12., 12.]),
        },
    },
    "1'E-2'2'": {
        "a": {
            "xp": np.cumsum([0.0, 2.0, 2.6, 1.6, 1.6, 1.6, 1.6,
                             3.9, 1.8, 2.3, 1.8, 2.0]),
            "p": np.array([11.25, 15.0, 15., 15., 15., 15.,
                           15., 15., 15., 15.]),
        },
    },
    "B'B'": {
        "a": {
            "xp": np.cumsum([0., 2.0, 3.0, 2.8, 3.0, 2.0]),
            "p": np.array([15., 15., 15., 15.]),
        },
        "b": {
            "xp": np.cumsum([0., 2.0, 3.2, 2.7, 3.2, 2.0]),
            "p": np.array([12., 12., 12., 12.]),
        },
    },
    "Bo'Bo'": {
        "a": {
            "xp": np.cumsum([0., 2.0, 3.0, 4.4, 3.0, 2.0]),
            "p": np.array([16., 16., 16., 16.]),
        },
        "b": {
            "xp": np.cumsum([0., 2.2, 3.2, 4.1, 3.2, 2.2]),
            "p": np.array([18., 18., 18., 18.]),
        },
        "c": {
            "xp": np.cumsum([0., 2.4, 2.6, 8.4, 2.6, 2.4]),
            "p": np.array([21., 21., 21., 21.]),
        },
        "d": {
            "xp": np.cumsum([0., 3.0, 2.4, 6.6, 2.4, 3.0]),
            "p": np.array([21., 21., 21., 21.]),
        },
        "e": {
            "xp": np.cumsum([0., 2.6, 2.7, 5.0, 2.7, 2.6]),
            "p": np.array([20., 20., 20., 20.]),
        },
        "f": {
            "xp": np.cumsum([0., 2.9, 2.6, 7.8, 2.6, 2.9]),
            "p": np.array([21., 21., 21., 21.]),
        },
    },
    "Co'Co'": {
        "a": {
            "xp": np.cumsum([0., 2.2, 2.0, 2.0, 6.3, 2.0, 2.0, 2.2]),
            "p": np.array([17., 17., 17., 17., 17., 17.]),
        },
        "b": {
            "xp": np.cumsum([0., 2.8, 1.8, 1.8, 4.8, 1.8, 1.8, 2.8]),
            "p": np.array([18., 18., 18., 18., 18., 18.]),
        },
        "c": {
            "xp": np.cumsum([0., 2.6, 1.8, 2.1, 7.9, 2.1, 1.8, 2.6]),
            "p": np.array([20., 20., 20., 20., 20., 20.]),
        },
        "d": {
            "xp": np.cumsum([0., 2.4, 1.8, 1.8, 11.0, 1.8, 1.8, 2.4]),
            "p": np.array([21., 21., 21., 21., 21., 21.]),
        },
        "e": {
            "xp": np.cumsum([0., 2.1, 2.0, 2.1, 9.0, 2.1, 2.0, 2.1]),
            "p": np.array([21., 21., 21., 21., 21., 21.]),
        },
        "f": {
            "xp": np.cumsum([0., 2.6, 1.8, 2.0, 7.9, 2.0, 1.8, 2.6]),
            "p": np.array([19., 19., 19., 19., 19., 19.]),
        },
    },
}


INFLUENCELINES = {
    "Hell": {
        "fx" : 10.,
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
        super(NorwegianLocomotive, self).__init__(loc['xp'], loc['p'])

    def todict(self):
        d = super(NorwegianLocomotive, self).todict()
        d["litra"] = self.litra
        d["sublitra"] = self.sublitra
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
        |   1    |     --1900 |    |     ls    | Local suburban train |
        |   2    | 1900--1930 |    |     p     | Passenger train      |
        |   3    | 1930--1930 |    |     f     | Freight train        |
        |   4    | 1960--1985 |    ====================================
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
        self.traintype = traintype
        locs, wags = [], []
        ttl = traintype.lower()
        if period == 1: # -- 1900
            if ttl == "ls" or ttl == "p":
                if ttl == "ls":
                    locs = [NorwegianLocomotive("1'C1't", "a")]
                elif ttl == "p":
                    locs = [NorwegianLocomotive("2'B-2", sublitra)
                            for sublitra in ["a", "b", "c"]]
                wags = [_load.BogieWagon(p, a, b, c, 5.0)
                        for p in [5.0, 9.0]
                        for a in [3.0, 3.2]
                        for b in [11.2, 11.9]
                        for c in [2.0, 2.1]]
            elif ttl == "f":
                locs = [NorwegianLocomotive("1'C-3", sublitra)
                        for sublitra in ["a", "b", "c"]]
                wags = (
                    [_load.TwoAxleWagon(p, a, b, 2.3)
                        for p in [2.3, 9.0]
                        for a in [1.5, 2.5]
                        for b in [2.5, 4.0]]
                  + [_load.BogieWagon(p, a, b, c, 2.3)
                        for p in [2.3, 9.0]
                        for a in [2.0, 2.6]
                        for b in [6.5, 11.0]
                        for c in [1.6]]
                )
        elif period == 2: # 1900 -- 1930
            if ttl == "ls" or ttl == "p":
                if ttl == "ls":
                    locs = [NorwegianLocomotive("1'C1't", "a")]
                elif ttl == "p":
                    locs = (
                        [NorwegianLocomotive("2'B-2", sublitra)
                         for sublitra in ["a", "b", "c"]]
                      + [NorwegianLocomotive(litra, sublitra)
                         for litra in ["2'C-2'2'", "2'D-2'2'"]
                         for sublitra in ["a", "b"]]
                    )
                wags = [_load.BogieWagon(p, a, b, c, 5.0)
                        for p in [5.0, 11.0]
                        for a in [2.4, 3.1]
                        for b in [11.6, 14.4]
                        for c in [1.9, 2.3]]
            elif ttl == "f":
                locs = (
                    [NorwegianLocomotive("1'C-3", sublitra)
                     for sublitra in ["a", "b", "c"]]
                  + [NorwegianLocomotive("1'D-2'2'", sublitra)
                     for sublitra in ["a", "b", "c", "d"]]
                )
                wags = (
                    [_load.TwoAxleWagon(p, a, b, 3.0)
                        for p in [3.0, 12.0]
                        for a in [1.5, 2.5]
                        for b in [2.5, 5.0]]
                  + [_load.BogieWagon(p, a, b, c, 3.0)
                        for p in [3.0, 12.0]
                        for a in [2.2, 2.7]
                        for b in [6.5, 11.5]
                        for c in [1.6, 1.9]]
                )
        elif period == 3: # 1930 -- 1960
            if ttl == "ls" or ttl == "p":
                if ttl == "ls":
                    xp = _load.get_geometry_bogie_wagon(3.6, 15.0, 2.5)
                    p = np.array([13.0]*4)
                    locs = [_load.Locomotive(xp, p)]
                elif ttl == "p":
                    locs = (
                        [NorwegianLocomotive(litra, sublitra)
                         for litra in ["2'C-2'2'", "2'D-2'2'", "B'B'"]
                         for sublitra in ["a", "b"]]
                    )
                wags = [_load.BogieWagon(p, a, b, c, 6.0)
                        for p in [6.0, 12.0]
                        for a in [1.9, 4.3]
                        for b in [9.1, 16.0]
                        for c in [2.0, 3.0]]
            elif ttl == "f":
                locs = (
                    [NorwegianLocomotive("1'E-2'2'", sublitra)
                     for sublitra in ["a"]]
                    + [NorwegianLocomotive("1'D-2'2'", sublitra)
                       for sublitra in ["a", "b", "c", "d"]]
                    + [NorwegianLocomotive("B'B'", sublitra)
                       for sublitra in ["a", "b"]]
                )

                wags = (
                    [_load.TwoAxleWagon(p, a, b, 3.0)
                        for p in [3.0, 15.0]
                        for a in [1.5, 3.0]
                        for b in [3.5, 7.0]]
                  + [_load.BogieWagon(p, a, b, c, 3.0)
                        for p in [3.0, 15.0]
                        for a in [2.4, 2.7]
                        for b in [8.0, 12.0]
                        for c in [2.0]]
                )
        elif period == 4: # 1960 -- 1985
            if ttl == "ls" or ttl == "p":
                if ttl == "ls":
                    xp = _load.get_geometry_bogie_wagon(3.8, 16.0, 2.5)
                    p = np.array([16.0]*4)
                    locs = [_load.Locomotive(xp, p),]
                elif ttl == "p":
                    locs = (
                        [NorwegianLocomotive(litra, sublitra)
                         for litra in ["Bo'Bo'", "Co'Co'", "B'B'"]
                         for sublitra in ["a", "b",]]
                    )
                wags = [_load.BogieWagon(p, a, b, c, 7.5)
                        for p in [7.5, 13.0]
                        for a in [3.0, 4.2]
                        for b in [16.0, 20.4]
                        for c in [2.2, 2.7]]
            elif ttl == "f":
                locs = ([NorwegianLocomotive(litra, sublitra)
                         for litra in ["Bo'Bo'", "Co'Co'", "B'B'"]
                         for sublitra in ["a", "b", ]]
                )
                wags = (
                    [_load.TwoAxleWagon(p, a, b, 5.0)
                        for p in [5.0, 18.0]
                        for a in [2.0, 4.1]
                        for b in [5.5, 9.0]]
                  + [_load.BogieWagon(p, a, b, c, 5.0)
                        for p in [5.0, 18.0]
                        for a in [2.5, 3.2]
                        for b in [9.0, 15.7]
                        for c in [1.8]]
                )
        elif period == 5: # 1985 -- 2000
            if ttl == "ls" or ttl == "p":
                if ttl == "ls":
                    xp = _load.get_geometry_bogie_wagon(3.8, 16.0, 2.6)
                    p = np.array([18.0]*4)
                    locs = [_load.Locomotive(xp, p),]
                elif ttl == "p":
                    locs = (
                        [NorwegianLocomotive(litra, sublitra)
                         for litra in ["Bo'Bo'", "Co'Co'"]
                         for sublitra in ["a", "b", "c", "d", "e", "f"]]
                    )
                wags = [_load.BogieWagon(p, a, b, c, 8.5)
                        for p in [8.5, 14.0]
                        for a in [3.2, 4.3]
                        for b in [16.5, 20.4]
                        for c in [2.5, 2.7]]
            elif ttl == "f":
                locs = ([NorwegianLocomotive(litra, sublitra)
                         for litra in ["Bo'Bo'", "Co'Co'"]
                         for sublitra in ["a", "b", "c", "d", "e", "f"]]
                )

                wags = (
                    [_load.TwoAxleWagon(p, a, b, 5.6)
                        for p in [5.6, 22.5]
                        for a in [2.3, 4.1]
                        for b in [7.5, 11.0]]
                  + [_load.BogieWagon(p, a, b, c, 5.6)
                        for p in [5.6, 22.5]
                        for a in [2.5]
                        for b in [9.0, 15.7]
                        for c in [1.8]]
                )
        elif period == 6: # 2000 --
            if ttl == "ls" or ttl == "p":
                if ttl == "ls":
                    xp = _load.get_geometry_bogie_wagon(3.8, 16.0, 2.6)
                    p = np.array([18.0]*4)
                    locs = [_load.Locomotive(xp, p),]
                elif ttl == "p":
                    locs = (
                        [NorwegianLocomotive(litra, sublitra)
                         for litra in ["Bo'Bo'", "Co'Co'"]
                         for sublitra in ["c", "d", "e", "f"]]
                    )
                wags = ([_load.BogieWagon(p, a, b, c, 8.5)
                        for p in [8.5, 14.0]
                        for a in [3.2, 4.3]
                        for b in [16.5, 20.4]
                        for c in [2.5, 2.7]]
                      + [_load.JacobsWagon(p, a, b, c, 8.5)
                        for p in [8.5, 14.0]
                        for a in [1.6, 5.6]
                        for b in [15.3, 18.2]
                        for c in [2.5]]
                )
            elif ttl == "f":
                locs = ([NorwegianLocomotive(litra, sublitra)
                         for litra in ["Bo'Bo'", "Co'Co'"]
                         for sublitra in ["c", "d", "e", "f"]]
                         )

                wags = (
                    [_load.TwoAxleWagon(p, a, b, 5.6)
                        for p in [5.6, 22.5]
                        for a in [2.3, 4.1]
                        for b in [7.5, 11.0]]
                    + [_load.BogieWagon(p, a, b, c, 5.6)
                        for p in [5.6, 22.5]
                        for a in [2.5]
                        for b in [9.0, 15.7]
                        for c in [1.8]]
                    + [_load.JacobsWagon(p, a, b, c, 5.6)
                        for p in [5.6, 22.5]
                        for a in [2.5, 2.8]
                        for b in [14.2, 14.9]
                        for c in [1.8]])
        super(NorwegianRollingStock, self).__init__(locs, wags)

    def todict(self):
        d = super(NorwegianRollingStock, self).todict()
        d["period"] = self.period
        d["traintype"] = self.traintype
        return d


class PacrilJSONDecoder(_serialize.PacrilJSONDecoder):
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
    for n in xrange(3):
        train = rs.get_train(np.random.randint(10, 50))
        l = INFLUENCELINES['Hell']['stringer']
        z = train.apply(l)
        plt.plot(z, label="Train {0}".format(n+1))
    plt.legend()
    plt.show(block=True)
