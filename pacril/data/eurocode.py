# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import numpy as np
from .. import _load


__all__ = ["ECTrainType{0:n}" for n in range(1, 13)] + ["ECTrainFactory"]


class BaseECTrain(_load.Train):
    @property
    def description(self):
        return self.__doc__


class ECTrainType1(BaseECTrain):
    """Locomotive-hauled passenger train"""
    def __init__(self):
        xp = np.cumsum([0., 1.4, 2.2, 2.2, 6.9, 2.2, 2.2, 1.4])
        p = np.array([22.5] * 6)
        loc = _load.Locomotive(xp, p)
        loc.speed = 200.0

        w = _load.BogieWagon(11.0, 1.8 + 2.6/2, 11.5 + 2.6, 2.6, 11.0)
        wagons = [w.copy() for n in range(12)]
        super(ECTrainType1, self).__init__(loc, wagons)


class ECTrainType2(BaseECTrain):
    """Locomotive-hauled passenger train"""
    def __init__(self):
        xp, p = np.cumsum([0., 1.4, 3.3, 6.7, 3.3, 1.4]), np.array([22.5] * 4)
        loc = _load.Locomotive(xp, p)
        loc.speed = 160.0

        w = _load.BogieWagon(11.0, 2.5 + 2.5 / 2, 16.5 + 2.5, 2.5, 11.0)
        wagons = [w.copy() for n in range(10)]

        super(ECTrainType2, self).__init__(loc, wagons)


class ECTrainType3(BaseECTrain):
    """High speed passenger train"""
    def __init__(self):
        xp = np.cumsum([0., 4.7, 3.0, 8.46, 3.0, 2.0])
        p = np.array([20.0] * 4)
        loc = _load.Locomotive(xp, p)
        loc.speed = 250.0

        A = _load.BogieWagon(15.0, 2.45 + 2.5 / 2, 16.5 + 2.5, 2.5, 15.0)
        xpB, pB = np.cumsum([0., 2.0, 3.0, 8.46, 3.0, 4.7]), np.array([20.0]*4)
        B = _load.Locomotive(xpB, pB)
        wseq = [A] * 13 + [B]
        wagons = [w.copy() for w in wseq]

        super(ECTrainType3, self).__init__(loc, wagons)


class ECTrainType4(BaseECTrain):
    """High speed passenger train"""
    def __init__(self):
        # Locomotive
        xp = np.cumsum([0., 3.5, 3.0, 11.0, 3.0, 1.65])
        p = np.array([17.0] * 4)
        loc = _load.Locomotive(xp, p)
        loc.speed = 250.0

        # Wagons
        xpA, pA = np.cumsum([0., 1.65, 3.0, 15.7, 1.5]), np.array([17.0] * 3)
        wA = _load.Load(xpA, pA)
        wA.pempty = wA.p

        xpB, pB = np.cumsum([0., 1.5, 15.7, 1.5]), np.array([17.0] * 2)
        wB = _load.Load(xpB, pB)
        wB.pempty = wB.p

        xpC, pC = np.cumsum([0., 1.5, 15.7, 3.0, 1.65]), np.array([17.0] * 3)
        wC = _load.Load(xpC, pC)
        wC.pempty = wC.p

        xpD = np.cumsum([0., 1.65, 3.0, 11.0, 3.0, 3.5])
        pD = np.array([17.0] * 4)
        wD = _load.Load(xpD, pD)
        wD.pempty = wD.p

        wseq = [wA] + [wB] * 8 + [wC] + [wD]
        wagons = [w.copy() for w in wseq]

        # Train
        super(ECTrainType4, self).__init__(loc, wagons)


class ECTrainType5(BaseECTrain):
    """Locomotive-hauled freight train"""
    def __init__(self):
        # Locomotive
        xp = np.cumsum([0., 2.0, 2.1, 2.1, 4.4, 2.1, 2.1, 2.0])
        p = np.array([22.5] * 6)
        loc = _load.Locomotive(xp, p)
        loc.speed = 100.0

        # Wagons
        xp = np.cumsum([0., 2.0, 1.8, 1.8, 5.7, 1.8, 1.8, 2.0])
        p = np.array([22.5] * 6)
        A = _load.Load(xp, p)
        A.pempty = A.p
        wagons = [A.copy() for w in range(15)]

        # Train
        super(ECTrainType5, self).__init__(loc, wagons)


class ECTrainType6(BaseECTrain):
    """Locomotive-hauled freight train"""
    def __init__(self):
        # Locomotive
        xp = np.cumsum([0., 2.0, 2.1, 2.1, 4.4, 2.1, 2.1, 2.0])
        p = np.array([22.5] * 6)
        loc = _load.Locomotive(xp, p)
        loc.speed = 100.0

        # Wagons
        A = _load.TwoAxleWagon(7.0, 1.9, 6.5, 22.5/4.)
        B = _load.BogieWagon(22.5, 1.8+1.8/2, 12.8+1.8, 1.8, 22.5/4)
        C = _load.BogieWagon(22.5, 1.6+1.8/2, 8.0+1.8, 1.8, 22.5/4)
        wseq = [A, A, B, A, C, C, A, B, B, B, A, A, C, C, A, B, C, A,
                A, C, C, B]
        wagons = [w.copy() for w in wseq]

        # Train
        super(ECTrainType6, self).__init__(loc, wagons)


class ECTrainType7(BaseECTrain):
    """Locomotive-hauled freight train"""
    def __init__(self):
        # Locomotive
        xp = np.cumsum([0., 1.4, 2.2, 2.2, 6.9, 2.2, 2.2, 1.4])
        p = np.array([22.5] * 6)
        loc = _load.Locomotive(xp, p)
        loc.speed = 80.0

        # Wagons
        wag = _load.BogieWagon(22.5, 1.6+1.8/2, 11+1.8, 1.8, 22.5 / 4.)
        wagons = [wag.copy() for n in range(10)]

        # Train
        super(ECTrainType7, self).__init__(loc, wagons)


class ECTrainType8(BaseECTrain):
    """Locomotive-hauled freight train"""
    def __init__(self):
        # Locomotive
        xp = np.cumsum([0., 1.4, 2.2, 2.2, 6.9, 2.2, 2.2, 1.4])
        p = np.array([22.5] * 6)
        loc = _load.Locomotive(xp, p)
        loc.speed = 100.0

        # Wagons
        wag = _load.TwoAxleWagon(22.5, 2.1, 5.5, 22.5/4.)
        wagons = [wag.copy() for n in range(20)]

        # Train
        super(ECTrainType8, self).__init__(loc, wagons)


class ECTrainType9(BaseECTrain):
    """Suburban multiple unit train"""
    def __init__(self):
        # Locomotive
        xp = np.cumsum([0., 2.15, 2.5, 14.0, 2.5, 2.15])
        p = np.array([13.0] * 4)
        loc = _load.Locomotive(xp, p)
        loc.speed = 120.0

        # Wagons
        wag1 = _load.BogieWagon(11.0, 2.15+2.5/2, 11.5+2.5, 2.5, 11.0 - 2.5)
        wag2 = _load.BogieWagon(13.0, 2.15+2.5/2, 14.0+2.5, 2.5, 13.0 - 2.5)
        wagons = [w.copy() for w in [wag1, wag2, wag2, wag1, wag2]]

        # Train
        super(ECTrainType9, self).__init__(loc, wagons)


class ECTrainType10(BaseECTrain):
    """Underground"""
    def __init__(self):
        # Wagons
        pa, pb = np.array([15.0, 10.0, 10.0, 15.0]), np.array([10.0]*4)
        A = _load.BogieWagon(pa, 1.75+2.4/2, 7.9+2.4, 2.4, pa - 2.5)
        B = _load.BogieWagon(pb, 1.75+2.4/2, 7.9+2.4, 2.4, pb - 2.5)
        wagons = [w.copy() for w in [B, B, A, A, B, B, A]]

        # Locomotive
        xp, p = A.xp, A.p
        loc = _load.Locomotive(xp, p)
        loc.speed = 120.0

        # Train
        super(ECTrainType10, self).__init__(loc, wagons)


class ECTrainType11(BaseECTrain):
    """Locomotive-hauled freight train"""
    def __init__(self):
        # Locomotive
        xp = np.cumsum([0., 1.4, 2.2, 2.2, 6.9, 2.2, 2.2, 1.4])
        p = np.array([22.5] * 6)
        loc = _load.Locomotive(xp, p)
        loc.speed = 120.0

        # Wagons
        wag = _load.BogieWagon(25.0, 1.5+2.0/2, 11.0+2.0, 2.0, 25.0/4.)
        wagons = [wag.copy() for n in range(10)]

        # Train
        super(ECTrainType11, self).__init__(loc, wagons)


class ECTrainType12(BaseECTrain):
    """Locomotive-hauled freight train"""
    def __init__(self):
        # Locomotive
        xp = np.cumsum([0., 1.4, 2.2, 2.2, 6.9, 2.2, 2.2, 1.4])
        p = np.array([22.5] * 6)
        loc = _load.Locomotive(xp, p)
        loc.speed = 100.0

        # Wagons
        wag = _load.TwoAxleWagon(25.0, 2.1, 5.5, 25.0/4.)
        wagons = [wag.copy() for n in range(20)]

        # Train
        super(ECTrainType12, self).__init__(loc, wagons)


def ECTrainFactory(type_number):
    """Factory for Eurocode real trains in Appendix D, EC1-2

    Returns a new instance of Eurocode real train with type
    `type_number`.

    Arguments
    ---------
    type_number : int

    Returns
    -------
    ECTrainType
        ECTrain train number as per Eurocode appendix D.

    """
    trains = {1: ECTrainType1, 2: ECTrainType2, 3: ECTrainType3,
              4: ECTrainType4, 5: ECTrainType5, 6: ECTrainType6,
              7: ECTrainType7, 8: ECTrainType8, 9: ECTrainType9,
              10: ECTrainType10, 11: ECTrainType11, 12: ECTrainType12}
    return trains[type_number]()
