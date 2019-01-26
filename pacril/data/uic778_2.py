# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import numpy as np
from .. import _load


__all__ = ["Train{0:n}" for n in range(1, 22)] + ["UICTrainFactory"]


class BaseUICTrain(_load.Train):
    pass


class UICTrain1(BaseUICTrain):
    """Passenger train"""
    def __init__(self):
        xp = np.cumsum([0., 2.42, 1.8, 2.60, 2.98, 1.65, 1.65, 1.74])
        p = np.array([13.5, 12.0, 12.0, 9.5, 9.5, 10.0])
        locomotive = _load.Locomotive(xp, p)
        locomotive.speed = None

        w = _load.TwoAxleWagon(5.0, 2.1, 4.0, 5.0)
        wagons = [w.copy() for n in range(6)]

        super(UICTrain1, self).__init__(locomotive, wagons)


class UICTrain2(BaseUICTrain):
    """Freight train"""
    def __init__(self):
        xp = np.cumsum([0., 2.97, 2.0, 1.4, 3.97, 1.65, 1.65, 1.74])
        p = np.array([13.5, 13.0, 14.0, 9.0, 9.5, 10.0])
        locomotive = _load.Locomotive(xp, p)
        locomotive.speed = None

        w1 = _load.TwoAxleWagon(8.0, 1.0, 3.5, 8.0)
        w2 = _load.TwoAxleWagon(7.0, 1.0, 2.8, 7.0)
        wseq = [w1] * 3 + [w2] * 2 + [w1] * 3 + [w2] * 4
        wagons = [w.copy() for w in wseq]

        super(UICTrain2, self).__init__(locomotive, wagons)


class UICTrain3(BaseUICTrain):
    """Passenger train"""
    def __init__(self):
        xp = np.cumsum([0., 2.49, 2.0, 2.5, 2.92, 1.65, 1.65, 1.74])
        p = np.array([14.0, 14.0, 14.0, 10.0, 12.0, 12.0])
        locomotive = _load.Locomotive(xp, p)
        locomotive.speed = None

        w = _load.ThreeAxleWagon(5.0, 1.85, 3.25, 5.0)
        wagons = [w.copy() for n in range(6)]

        super(UICTrain3, self).__init__(locomotive, wagons)


class UICTrain4(BaseUICTrain):
    """Freight train"""
    def __init__(self):
        xp = np.cumsum([0., 2.97, 2.0, 1.4, 3.97, 1.65, 1.65, 1.74])
        p = np.array([13.5, 13.0, 14.0, 9.0, 9.5, 10.0])
        locomotive = _load.Locomotive(xp, p)
        locomotive.speed = None

        w1 = _load.TwoAxleWagon(12.0, 1.4, 4.0, 12.0)
        w2 = _load.TwoAxleWagon(8.0, 1.4, 4.0, 8.0)
        wsec = [w1, w1, w2] + [w1] * 6 + [w2] * 3
        wagons = [w.copy() for w in wsec]

        super(UICTrain4, self).__init__(locomotive, wagons)


class UICTrain5(BaseUICTrain):
    """Passenger train"""
    def __init__(self):
        xp = np.cumsum([0., 1.45, 2.2, 2.6, 2.6, 2.62, 1.55, 1.6, 1.55, 1.45])
        p = np.array([10.5, 10.5, 15.0, 15.0, 11.0, 11.0, 11.0, 11.0])
        locomotive = _load.Locomotive(xp, p)
        locomotive.speed = None

        w1 = _load.TwoAxleWagon(6.8, 3.1, 6.0, 6.8)
        w2 = _load.ThreeAxleWagon(7.7, 2.2, 4.6, 7.7)
        wsec = [w1] + [w2] * 3 + [w1] * 2
        wagons = [w.copy() for w in wsec]

        super(UICTrain5, self).__init__(locomotive, wagons)


class UICTrain6(BaseUICTrain):
    """Express train"""
    def __init__(self):
        xp = np.cumsum([0., 1.45, 2.2, 2.6, 2.6, 2.52, 1.55, 1.5, 1.55, 1.6])
        p = np.array([11.5, 11.0, 16.0, 16.5, 11.0, 11.5, 13.5, 13.5])
        locomotive = _load.Locomotive(xp, p)
        locomotive.speed = None

        w = _load.BogieWagon(10.6, 2.0 + 2.5/2, 10.7 + 2.5, 2.5, 10.6)
        wagons = [w.copy() for n in range(6)]

        super(UICTrain6, self).__init__(locomotive, wagons)


class UICTrain7(BaseUICTrain):
    """Freight train"""
    def __init__(self):
        xp = np.cumsum([0., 1.64, 2.3, 2.0, 2.0, 3.01, 1.65, 1.65, 1.74])
        p = np.array([9.0, 12.5, 13.5, 13.0, 10.0, 12.0, 12.0])
        locomotive = _load.Locomotive(xp, p)
        locomotive.speed = None

        w1 = _load.TwoAxleWagon(8.0, 1.4, 4.0, 8.0)
        w2 = _load.TwoAxleWagon(12.0, 1.4, 4.0, 12.0)
        wsec = [w1, w2, w1, w1] + [w2] * 6 + [w1] * 5
        wagons = [w.copy() for w in wsec]

        super(UICTrain7, self).__init__(locomotive, wagons)


class UICTrain8(BaseUICTrain):
    """Passenger train"""
    def __init__(self):
        xp = np.cumsum([0., 1.85, 2.8, 2.0, 2.0, 2.0, 2.8, 2.1, 1.8, 2.0, 1.8])
        p = np.array([17.5, 19.0, 19.5, 18.5, 19.0, 16.0, 16.0, 16.0, 16.0])
        locomotive = _load.Locomotive(xp, p)
        locomotive.speed = None

        w1 = _load.TwoAxleWagon(10.0, 1.6, 4.5, 10.0)
        w2 = _load.TwoAxleWagon(11.0, 1.8, 7.0, 11.0)
        wsec = [w1] * 2 + [w2] + [w1] * 2 + [w2]
        wagons = [w.copy() for w in wsec]

        super(UICTrain8, self).__init__(locomotive, wagons)


class UICTrain9(BaseUICTrain):
    """Express train"""
    def __init__(self):
        xp = np.cumsum([0., 1.62, 2.2, 2.2, 2.1, 2.6, 2.97, 1.8, 2.0, 1.8,
                        1.83])
        p = np.array([15.0, 15.0, 17.5, 18.0, 18.0, 16.0, 16.0, 16.5, 16.5])
        locomotive = _load.Locomotive(xp, p)
        locomotive.speed = None

        w = _load.BogieWagon(11.5, 2.92 + 2.15 / 2, 11.58 + 2.15, 2.15, 11.5)
        wagons = [w.copy() for n in range(7)]

        super(UICTrain9, self).__init__(locomotive, wagons)


class UICTrain10(BaseUICTrain):
    """Freight train"""
    def __init__(self):
        xp = np.cumsum([0., 1.5, 2.5, 1.5, 1.5, 1.5, 1.5, 3.0, 2.4, 1.5, 1.5])
        p = np.array([13.0, 16.5, 16.0, 16.5, 16.5, 16.0, 15.0, 15.0, 15.0])
        locomotive = _load.Locomotive(xp, p)
        locomotive.speed = None

        w1 = _load.TwoAxleWagon(12.0, 1.6, 4.5, 12.0)
        w2 = _load.TwoAxleWagon(13.0, 1.8, 7.0, 13.0)
        w3 = _load.BogieWagon(13.0, 1.6 + 2.0 / 2, 10.8 + 2.0, 2.0, 13.0)
        w4 = _load.TwoAxleWagon(6.0, 1.6, 4.5, 6.0)
        w5 = _load.TwoAxleWagon(6.5, 1.8, 7.0, 6.5)
        wsec = [w1, w2, w3, w3, w2, w2, w2, w1, w1, w4, w4, w4, w2, w2, w1, w1,
                w5, w5, w3, w3]
        wagons = [w.copy() for w in wsec]

        super(UICTrain10, self).__init__(locomotive, wagons)


class UICTrain11(BaseUICTrain):
    """Passenger train"""
    def __init__(self):
        xp = np.cumsum([0., 1.5, 2.2, 2.0, 2.0, 2.0, 2.5, 1.8, 1.8, 1.8, 1.5])
        p = np.array([16.0, 19.0, 19.0, 19.0, 19.0, 16.0, 16.0, 16.0, 16.0])
        locomotive = _load.Locomotive(xp, p)
        locomotive.speed = None

        w1 = _load.TwoAxleWagon(12.5, 2.71, 8.5, 12.5)
        w2 = _load.ThreeAxleWagon(7.0, 2.57, 3.75, 7.0)
        wsec = [w1, w2, w2, w2, w1, w1]
        wagons = [w.copy() for w in wsec]

        super(UICTrain11, self).__init__(locomotive, wagons)


class UICTrain12(BaseUICTrain):
    """Express train"""
    def __init__(self):
        xp = np.cumsum([0., 1.5, 2.2, 2.0, 2.0, 2.0, 2.5, 1.8, 1.8, 1.8, 1.5])
        p = np.array([16.0, 19.0, 19.0, 19.0, 19.0, 16.0, 16.0, 16.0, 16.0])
        locomotive = _load.Locomotive(xp, p)
        locomotive.speed = None

        w = _load.BogieWagon(11.5, 2.92 + 2.15 / 2, 11.58 + 2.15, 2.15, 11.5)
        wagons = [w.copy() for n in range(7)]

        super(UICTrain12, self).__init__(locomotive, wagons)


class UICTrain13(BaseUICTrain):
    """Freight train"""
    def __init__(self):
        xp = np.cumsum([0., 1.5, 2.2, 1.5, 1.5, 1.5, 1.5, 3.0, 2.5, 1.5, 1.5])
        p = np.array([15.0] + [18.0] * 5 + [16.0] * 3)
        locomotive = _load.Locomotive(xp, p)
        locomotive.speed = None

        w1 = _load.TwoAxleWagon(13.0, 1.6, 4.5, 13.0)
        w2 = _load.TwoAxleWagon(13.0, 1.8, 7.0, 13.0)
        w3 = _load.BogieWagon(10.0, 1.6 + 2.0 / 2, 10.8 + 2.0, 2.0, 10.0)
        w4 = _load.TwoAxleWagon(9.0, 1.6, 4.5, 9.0)
        w5 = w1.copy()
        w5.p, w5.pempty = w5.p * .5, w5.pempty * .5
        w6 = w2.copy()
        w6.p, w6.pempty = w6.p * .5, w6.pempty * .5
        w7 = w3.copy()
        w7.p, w7.pempty = w7.p * .5, w7.pempty * .5
        wsec = ([w1, w2, w3] + [w4] * 3 + [w2, w1, w1, w3, w2, w5, w5, w6]
                + [w2, w1] + [w4] * 3 + [w3] * 2 + [w7] + [w2, w4])
        wagons = [w.copy() for w in wsec]

        super(UICTrain13, self).__init__(locomotive, wagons)


class UICTrain14(BaseUICTrain):
    """Motive power unit"""
    def __init__(self):
        xp = np.cumsum([0., 3.375, 6.0, 3.375])
        p = np.array([8.0, 8.0])
        locomotive = _load.Locomotive(xp, p)
        locomotive.speed = None

        w1 = _load.TwoAxleWagon(5.0, 3.0, 4.5, 5.0)
        w2 = _load.TwoAxleWagon(8.0, 3.375, 6.0, 8.0)
        wagons = [w.copy() for w in [w1, w2, w1]]

        super(UICTrain14, self).__init__(locomotive, wagons)


class UICTrain15(BaseUICTrain):
    """Passenger train"""
    def __init__(self):
        xp = np.cumsum([0., 1.5, 2.2, 2.0, 2.0, 2.0, 2.5, 1.8, 1.8, 1.8, 1.5])
        p = np.array([18.0] + [20.0] * 4 + [17.0] * 4)
        locomotive = _load.Locomotive(xp, p)
        locomotive.speed = None

        w = _load.BogieWagon(9.0, 2.45 + 2.5 / 2, 16.5 + 2.5, 2.5, 9.0)
        wagons = [w.copy() for n in range(5)]

        super(UICTrain15, self).__init__(locomotive, wagons)


class UICTrain16(BaseUICTrain):
    """Express train"""
    def __init__(self):
        xp = np.cumsum([0., 1.5, 2.2, 2.0, 2.0, 2.0, 2.5, 1.8, 1.8, 1.8, 1.5])
        p = np.array([18.0] + [20.0] * 4 + [17.0] * 4)
        locomotive = _load.Locomotive(xp, p)
        locomotive.speed = None

        w = _load.BogieWagon(10.0, 2.45 + 2.5 / 2., 16.5 + 2.5, 2.5, 10.0)
        wagons = [w.copy() for n in range(6)]

        super(UICTrain16, self).__init__(locomotive, wagons)


class UICTrain17(BaseUICTrain):
    """Freight train"""
    def __init__(self):
        xp = np.cumsum([0., 1.95, 3.0, 1.85, 1.85, 1.85, 1.85, 3.20, 2.18,
                        1.75, 1.5, 1.37, 1.37, 1.92])
        p = np.array([12.5] + [20.0] * 5 + [14.5] * 6)
        locomotive = _load.Locomotive(xp, p)
        locomotive.speed = None

        w1 = _load.TwoAxleWagon(15.0, 2.16, 4.68, 15.0)
        w2 = _load.TwoAxleWagon(16.0, 2.55, 5.4, 16.0)
        w3 = _load.BogieWagon(14.0, 1.7 + 2.0 / 2, 6.9 + 2.0, 2.0, 14.0)
        w4 = w1.copy()
        w4.p, w4.pempty = w4.p * .6, w4.pempty * .6
        w5 = w3.copy()
        w5.p, w5.pempty = w5.p * 1.2, w5.pempty * 1.2
        w6 = w2.copy()
        w6.p, w6.pempty = w6.p * .4, w6.pempty * .4
        w7 = w3.copy()
        w7.p, w7.pempty = w7.p * .4, w7.pempty * .4
        wsec = ([w1, w2, w3, w2, w2, w1, w1, w4, w4, w3, w5, w5, w6]
                + [w2] * 3 + [w1, w3, w7, w3, w1])
        wagons = [w.copy() for w in wsec]

        super(UICTrain17, self).__init__(locomotive, wagons)


class UICTrain18(BaseUICTrain):
    """Passenger train"""
    def __init__(self):
        xp = np.cumsum([0., 2.6, 3.4, 4.5, 3.4, 2.6])
        p = np.array([20.0] * 4)
        locomotive = _load.Locomotive(xp, p)
        locomotive.speed = None

        w = _load.BogieWagon(9.0, 2.5 + 2.5 / 2, 16.5 + 2.5, 2.5, 9.0)
        wagons = [w.copy() for n in range(5)]

        super(UICTrain18, self).__init__(locomotive, wagons)


class UICTrain19(BaseUICTrain):
    """Express train"""
    def __init__(self):
        xp = np.cumsum([0., 2.7, 2.3, 2.3, 5.1, 2.3, 2.3, 2.7])
        p = np.array([19.5] * 6)
        locomotive = _load.Locomotive(xp, p)
        locomotive.speed = None

        w = _load.BogieWagon(11.0, 2.5 + 2.5 / 2, 16.5 + 2.5, 2.5, 11.0)
        wagons = [w.copy() for n in range(8)]

        super(UICTrain19, self).__init__(locomotive, wagons)


class UICTrain20(BaseUICTrain):
    """Freight train"""
    def __init__(self):
        xp = np.cumsum([0., 2.6, 3.4, 4.5, 3.4, 2.6])
        p = np.array([21.5] * 4)
        locomotive = _load.Locomotive(xp, p)
        locomotive.speed = None

        A1 = _load.TwoAxleWagon(15.0, 2.55, 5.4, 15.0)
        B = _load.TwoAxleWagon(14.0, 2.16, 4.68, 14.0)
        C = _load.TwoAxleWagon(15.0, 2.85, 6.8, 15.0)
        D = _load.TwoAxleWagon(6.5, 2.16, 4.68, 6.5)
        E1 = _load.BogieWagon(16.0, 1.7 + 2.0 / 2, 6.9 + 2.0, 2.0, 16.0)
        E2 = E1.copy()
        E2.p, E2.pempty = E2.p*.5, E2.pempty * .5
        A2 = A1.copy()
        A2.p, A2.pempty = A2.p * .5, A2.pempty * .5
        wsec = ([A1, B, C, D, E1, E1, D, D, A1, A1, C, B, B, D, D, E2, E2,
                 B, B, D, B, D, E1, A2, A1, B, B, E1, B])
        wagons = [w.copy() for w in wsec]

        super(UICTrain20, self).__init__(locomotive, wagons)


class UICTrain21(BaseUICTrain):
    """Freight train"""
    def __init__(self):
        xp = np.cumsum([0., 2.55, 1.85, 1.85, 8.3, 1.85, 1.85, 2.55])
        p = np.array([20.5] * 6)
        locomotive = _load.Locomotive(xp, p)
        locomotive.speed = None

        w = _load.BogieWagon(20.0, 2.08 + 2.0 / 2, 4.7 + 2.0, 2.0, 20.0)
        wagons = [w.copy() for n in range(25)]

        super(UICTrain21, self).__init__(locomotive, wagons)


def UICTrainFactory(train_number):
    """Factory for UIC trains

    Returns a new instance of UIC train `train_number`

    Arguments
    ---------
    train_number : int

    Returns
    -------
    UICTrain
        UIC train number as per Annexe 1 of Appendix 2 in UIC 778-2.
    """
    trains = {
        1: UICTrain1, 2: UICTrain2, 3: UICTrain3, 4: UICTrain4,
        5: UICTrain5, 6: UICTrain6, 7: UICTrain7, 8: UICTrain8,
        9: UICTrain9, 10: UICTrain10, 11: UICTrain11, 12: UICTrain12,
        13: UICTrain13, 14: UICTrain14, 15: UICTrain15, 16: UICTrain16,
        17: UICTrain17, 18: UICTrain18, 19: UICTrain19, 20: UICTrain20,
        21: UICTrain21}
    return trains[train_number]()
