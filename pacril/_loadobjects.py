# -*- coding: utf-8 -*-
import numpy as np
import copy
import unittest
from _load import *
from data import LOCOMOTIVES
import logging

logging.basicConfig(level=logging.DEBUG)

debug = logging.debug


class BaseLoad(object):
    """Parent object of load objects, all other loadobjects are children of
    this class

    """
    def __init__(self):
        self.fx = 10.

    def copy(self):
        return copy.deepcopy(self)

    def _set_loadvector(self):
        try:
            self.loadvector = get_loadvector(self.p, self.xp, self.fx)
        except AttributeError:
            pass

    def apply(self, influence_line):
        """Apply the load to the influence line to obtain the response

        Arguments
        ---------
        influence_line : ndarray
            The influence line to apply the load to.

        Returns
        -------
        ndarray
            The response of the signal after applying the load
        """
        f = self.loadvector
        return np.convolve(f, l, mode='full')

    def __setattr__(self, name, value):
        super(BaseLoad, self).__setattr__(name, value)
        if (name=="xp") | (name=="p") | (name=="fx"):
            self._set_loadvector()


class Load(BaseLoad):
    """Define a generic load object by loadmagnitude p and geometry vector xp.

    The load is defined by magnitudes p and geometry xp, see figure below.

        Loads:          p1  p2      p3  p4
                         |  |        |  |
                         |  |        |  |
                         V  V        V  V
        Geometry:   |----|--|--------|--|----|----->
                    x0  x1  x2       x3 x4   x5    xp
                    ^                        ^
                    |  <=== direction <===   |
              start of load             end of load

    Arguments
    ---------
    p : float or ndarray
        Defines the load magnitude, if p is a float it assumes equal load at
        all load positions, if p is an ndarray it defines each load magnitude
        individually.

    xp : ndarray
        Defines the geometry of the load, i.e the start, end and load
        positions. The array also defines the number of loads,
        i.e (n_loads = xp.size-2).
    """
    def __init__(self, xp, p):
        super(Load, self).__init__()
        self.xp = xp
        self.p = p

    @property
    def nloads(self):
        return len(self.xp)-2


class BaseVehicle(BaseLoad):
    """Parent object of vehicles, extends the definition of load to contain
    a vehicle properties and methods such as definition of axles and goods.

    """
    @property
    def naxles(self):
        return len(self.xp)-2

    def _set_goods_transported(self):
        try:
            self.goods_transported = (self.p - self.pempty).sum()
        except AttributeError:
            pass

    def _get_axleloadvector(self, p):
        if isinstance(p, float):
            return np.array([p] * self.naxles)
        else:
            return np.array(p)

    def __setattr__(self, name, value):
        if (name == 'p') | (name == 'pempty'):
            value = self._get_axleloadvector(value)
            super(BaseVehicle, self).__setattr__(name, value)
            self._set_goods_transported()
        else:
            super(BaseVehicle, self).__setattr__(name, value)


class Locomotive(BaseVehicle):
    """Data and functionality to generate loads from locomotives

    Interfaces to the locomotives defined in the data modules of the pacril
    package.

    Arguments
    ---------
    litra, sublitra: str
        The litra (e.g "B'B'") and sublitra (e.g "a") for the locomotives
        defined in the data module
    """
    def __init__(self, litra, sublitra):
        super(Locomotive, self).__init__()
        self.litra = litra
        self.sublitra = sublitra
        loc = LOCOMOTIVES[litra][sublitra]
        self.xp = loc['xp']
        self.p = loc['p']
        self.pempty = loc['p']


class TwoAxleWagon(BaseVehicle):
    """Define a twoaxle wagon

     The two axle wagon is defined by axleloads p and geometry parameter a and
     b, see figure below.

        Type:          Twoaxle wagon
                       +------------+
        Axleload:      | p1      p2 |
                    (--+------------+--)
                         O        O
        Geometry:   |----|--------|----|
                      a       b     a

    Arguments
    ---------
    p, pempty : float or ndarray
        Defines the axle loads, if float it assumes equal axle load on each
        axle, if ndarray it defines each axle load individually.

    a,b : float
        Defines the geometry of the vehicle, i.e the start, end and axle
        positions.
    """
    def __init__(self, p, a, b, pempty):
        super(TwoAxleWagon, self).__init__()
        self.xp = get_geometry_twoaxle_wagon(a, b)
        self.p = p
        self.pempty = pempty


class BogieWagon(BaseVehicle):
    """Define a bogie wagon

    The bogie wagon is defined by axleloads p and geometry parameters a, b and
    c, see figure below.

        Type:               Bogie wagon
                       +-------------------+
        Axleload:      | p1 p2       p3 p4 |
                    (--+-------------------+--)
                         O   O       O   O
                    |------|-----------|------|
        Geometry:      a         b         a
                         |---|       |---|
                           c           c

    Arguments
    ---------
    p, pempty : float or ndarray
        Defines the axle loads, if float it assumes equal axle load on each
        axle, if ndarray it defines each axle load individually.

    a,b,c : float
        Defines the geometry of the vehicle, i.e the start, end and axle
        positions.
    """
    def __init__(self, p, a, b, c, pempty):
        super(BogieWagon, self).__init__()
        self.xp = get_geometry_bogie_wagon(a, b, c)
        self.p = p
        self.pempty = pempty


class JacobsWagon(BaseVehicle):
    """Define a jacobs wagon

    The jacobsbogie wagon is defined by axleloads p and geometry parameters a,
    b and c, see figure below.

        Type:                     Jacobsbogie wagon
                       +----------------------------------+
        Axleload:      | p1 p2        p3  p4        p5 p6 |
                    (--+----------------)(----------------+--)
                         O   O        O    O        O   O
                    |------|------------|-------------|------|
        Geometry:      a         b             b          a
                         |---|        |----|        |---|
                           c            c             c

    Arguments
    ---------
    p, pempty : float or ndarray
        Defines the axle loads, if float it assumes equal axle load on each
        axle, if ndarray it defines each axle load individually.

    a,b,c : float
        Defines the geometry of the vehicle, i.e the start, end and axle
        positions.
    """
    def __init__(self, p, a, b, c, pempty):
        super(JacobsWagon, self).__init__()
        self.xp = get_geometry_jacobs_wagon(a, b, c)
        self.p = p
        self.pempty = pempty


class TestLocomotive(unittest.TestCase):
    def setUp(self):
        self.load = Locomotive("Bo'Bo'", "b")
        self.xptrue = np.array([0., 2.2, 5.4, 9.5, 12.7, 14.9])
        self.ptrue = np.array([18., 18., 18., 18.])
        self.naxles = 4
        self.goods_transported = 0.
        self.loadvector = np.array([
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 18., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 18., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 18., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 18., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
            ])

    def test_xp(self):
        np.testing.assert_allclose(self.load.xp, self.xptrue)

    def test_p(self):
        np.testing.assert_allclose(self.load.p, self.ptrue)

    def test_naxles(self):
        np.testing.assert_equal(self.load.naxles, self.naxles)

    def test_goods_transported(self):
        self.assertEqual(self.load.goods_transported, self.goods_transported)

    def test_loadvector(self):
        np.testing.assert_allclose(self.load.loadvector, self.loadvector)
        fx = self.load.fx
        self.load.fx = 100.
        self.load.fx = fx
        np.testing.assert_allclose(self.load.loadvector, self.loadvector)


class TestTwoAxleWagon(TestLocomotive):
    def setUp(self):
        self.load = TwoAxleWagon(np.array([13., 21.]), 4., 9.,
                                 np.array([1., 3.]))
        self.xptrue = np.cumsum([0., 4., 9., 4.])
        self.ptrue = np.array([13., 21.])
        self.naxles = 2
        self.goods_transported = 13.+21.-1.-3.
        self.loadvector = np.array([
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 13., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 21., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0.,
            ])


class TestBogieWagon(TestLocomotive):
    def setUp(self):
        self.load = BogieWagon(5., 1., 2., 1., 4.)
        self.xptrue = np.array([0., 0.5, 1.5, 2.5, 3.5, 4.])
        self.ptrue = np.array([5., 5., 5., 5.])
        self.naxles = 4
        self.goods_transported = 4. * (5.-4.)
        self.loadvector = np.array([
            0.,  0.,  0.,  0.,  0.,  5.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
            0.,  0.,  5.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  5.,
            0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  5.,  0.,  0.,  0.,
            0.,  0.])


class TestJacobsWagon(TestLocomotive):
    def setUp(self):
        p = np.array([9., 9., 13., 14., 10., 11.])
        self.load = JacobsWagon(p, 1., 2., 1., 7.)
        self.xptrue = np.array([0., 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.])
        self.ptrue = p
        self.naxles = 6
        self.goods_transported = np.sum(p) - 6. * 7
        self.loadvector = np.array([
            0.0, 0.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 13.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 14.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 11.0, 0.0, 0.0, 0.0, 0.0, 0.0])


if __name__ == '__main__':
    unittest.main()
    # from  _influence_line import get_il_simply_supported_beam
    # l  = get_il_simply_supported_beam(13., .5)
    # loc = Locomotive("2'C-2'2'", "a")
    # z = loc.apply(l)


    # import matplotlib.pyplot as plt
    # plt.plot(z)
    # plt.show(block=True)