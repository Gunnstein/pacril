# -*- coding: utf-8 -*-
import numpy as np
import copy
import unittest
from _load import *


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
    """Define a generic locomotive, see also NorwegianLocomotive.

    The locmomotive is defined by magnitudes p and geometry xp, see figure
    below.

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
        super(Locomotive, self).__init__()
        self.xp = xp
        self.p = p
        self.pempty = p


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


class Train(BaseVehicle):
    """Defines a train

    A train consists of a locomotive and wagons.

    Arguments
    ---------
    locomotive : Locomotive
        The locomotive of the train

    wagons : list
        A list whos elements are Wagon instances.

    Example
    -------
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> import pacril
        >>>
        >>> loc = pacril.NorwegianLocomotive("Bo'Bo'", "a")
        >>> wagons = [pacril.TwoAxleWagon(p, 3., 7., 5.)
                      for p in np.arange(10., 21., 1.)]
        >>> train = pacril.Train(loc, wagons)
        >>>
        >>> l = pacril.get_il_simply_supported_beam(5., .5)
        >>> z = train.apply(l)
        >>> xz = pacril.get_coordinate_vector(z)
        >>>
        >>> plt.plot(xz, z)
        >>> plt.show()
    """
    def __init__(self, locomotive, wagons):
        super(Train, self).__init__()
        self.locomotive = locomotive
        self.wagons = wagons
        self._set_xp_and_p()

    def _set_xp_and_p(self):
        dxp = np.diff(self.locomotive.xp)
        pitch = dxp[:-1]
        x0 = dxp[-1]
        p = self.locomotive.p
        pempty = self.locomotive.pempty
        if self.nwagons > 0:
            for wagon in self.wagons:
                dxp = np.diff(wagon.xp)
                dxp[0] = dxp[0] + x0
                pitch = np.concatenate((pitch, dxp[:-1]))
                x0 = dxp[-1]
                p = np.concatenate((p, wagon.p))
                pempty = np.concatenate((pempty, wagon.pempty))
        pitch = np.concatenate((pitch, np.array([x0])))
        xp = np.insert(np.cumsum(pitch), 0, 0.)
        for d in ["xp", "p", "pempty"]:
            try:
                del self.__dict__[d]
            except KeyError:
                pass
        self.xp, self.p, self.pempty = xp, p, pempty

    @property
    def nwagons(self):
        return len(self.wagons)

    def swap_locomotive(self, locomotive):
        """Swap out the locomotive

        Arguments
        ---------
        locomotive : Locomotive
            The locmotive to insert.
        """
        self.locomotive = locomotive
        self._set_xp_and_p()

    def swap_wagon(self, ix, wagon):
        """Swap out wagon with new wagon

        Arguments
        ---------
        ix : int
            The index of the wagon to swap out

        wagon : Wagon
            The wagon to swap in.
        """
        self.wagons[ix] = wagon
        self._set_xp_and_p()

    def insert_wagon(self, ix, wagon):
        """Insert a new wagon before ix

        Arguments
        ---------
        ix : int
            The index to insert the wagon infront of.

        wagon : Wagon
            The wagon to insert.
        """
        self.wagons.insert(ix, wagon)
        self._set_xp_and_p()

    def remove_wagon(self, ix):
        """Remove wagon from train

        Arguments
        ---------
        ix : int
            The index of the wagon to remove
        """
        self.wagons.pop(ix)
        self._set_xp_and_p()


class RollingStock(object):
    """Rolling stock contains the possible sets of locomotives and wagons.

    A rolling stock object contains locomotives and wagons and methods to
    generate and assemble trains.

    TODO: Create a unit test for this object.

    Arguments
    ---------
    locomotives, wagons : list
        A list of locomotives and wagons that are present in the rolling stock.
    loc_pmf, wagon_pmf : ndarray
        The probability mass funcitons (probabilities) of selecting each of
        the locomotives. Same size as locomotives/wagons.
    """
    def __init__(self, locomotives, wagons, loc_pmf=None, wagon_pmf=None):
        self.locomotives = locomotives
        self.wagons = wagons
        Nloc, Nwag = len(locomotives), len(wagons)
        self.locomotive_pmf = loc_pmf
        self.wagon_pmf = wagon_pmf

    @property
    def nlocomotives(self):
        return len(self.locomotives)

    @property
    def nwagons(self):
        return len(self.wagons)

    def choose_locomotive(self):
        if self.nlocomotives == 0:
            return []
        else:
            return np.random.choice(self.locomotives, p=self.locomotive_pmf)

    def choose_wagons(self, num_wagons):
        if self.nwagons == 0:
            return []
        else:
            return list(np.random.choice(
                self.wagons, size=num_wagons, p=self.wagon_pmf))

    def get_train(self, num_wagons):
        """Returns a train instance assembled randomly from the rolling stock.

        Arguments
        ---------
        num_wagons : int
            The number of wagons to assemble the train with.
        """
        loc = self.choose_locomotive()
        wagons = self.choose_wagons(num_wagons)
        return Train(loc, wagons)

    def get_neighbor_train(self, train, fixed_length_trains=True, Nwag_min=10,
                           Nwag_max=50):
        """Returns the neighbor train.

        The neighbor train is defined as the train that has one by adding,
        removing wagons or swapping the locomotive or a wagon.

        This function is very useful in simulated annealing where the
        neighboring solution can be determined by swapping out one of the
        elements.

        Arguments
        ---------
        train : Train
            The train instance to find the neighbor for.
        fixed_length_trains : bool
            Wether or not the new train should have the same number of wagons
            as the previous train or not.
        """
        Nwag = train.nwagons
        train_new = train.copy()

        if fixed_length_trains:
            n = np.random.randint(-1, Nwag)
        else:
            if Nwag_min < Nwag < Nwag_max:
                n = np.random.randint(-3, Nwag)
            elif Nwag == Nwag_min:
                n = np.random.randint(-2, Nwag)
            elif Nwag == Nwag_max:
                n = np.random.randint(-2, Nwag)
                if n == -2:
                    n = -3
        if n == -3:
            n = np.random.randint(0, Nwag)
            train_new.remove_wagon(n)
        if n == -2:
            n = np.random.randint(0, Nwag)
            train_new.insert(n, self.choose_wagons(1)[0])
        elif n == -1:
            train_new.swap_locomotive(self.choose_locomotive())
        else:
            train_new.swap_wagon(n, self.choose_wagons(1)[0])
        return train_new


class TestLocomotive(unittest.TestCase):
    def setUp(self):
        self.xptrue = np.array([0., 2.2, 5.4, 9.5, 12.7, 14.9])
        self.ptrue = np.array([18., 18., 18., 18.])
        self.load = Locomotive(self.xptrue.copy(), self.ptrue.copy())

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


class TestTrain(TestLocomotive):
    def setUp(self):
        loc = Locomotive(np.array([0., 2., 4., 6.]), np.array([1., 1.]))
        wagons = [TwoAxleWagon(2., 2., 4., 1.)]
        self.load = Train(loc, wagons)
        self.xptrue = np.array([0., 2., 4., 8., 12., 14.])
        self.ptrue = np.array([1., 1., 2., 2.])
        self.naxles = 4
        self.goods_transported = 2.*(2.-1.)
        self.loadvector = np.array([
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 2., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0.])

    def test_remove_and_insert_wagon(self):
        loc = self.load.locomotive.copy()
        wagons = copy.deepcopy(self.load.wagons)
        self.load.remove_wagon(0)
        np.testing.assert_raises(
            AssertionError, np.testing.assert_array_equal,
            self.load.loadvector, self.loadvector)
        self.load.insert_wagon(0, wagons[0])
        np.testing.assert_array_equal(self.loadvector, self.loadvector)


class TestRollingStock(TestLocomotive):
    def setUp(self):
        xploc = np.array([0., 2.2, 5.4, 9.5, 12.7, 14.9])
        ploc = np.array([18., 18., 18., 18.])
        locs = [Locomotive(xploc, ploc)]
        wags = [TwoAxleWagon(4., 1., 2., 2.)]
        self.rs = RollingStock(locs, wags)
        self.load = self.rs.get_train(15)

        self.traintrue = Train(locs[0], wags*15)
        self.xptrue = self.traintrue.xp
        self.ptrue = self.traintrue.p
        self.naxles = self.traintrue.naxles
        self.goods_transported = self.traintrue.goods_transported
        self.loadvector = self.traintrue.loadvector
        self.nwagontrue = self.traintrue.nwagons


    # def test_train_loadvector


if __name__ == '__main__':
    unittest.main()
    # from data import NorwegianLocomotive
    # import matplotlib.pyplot as plt
    # locs = [NorwegianLocomotive(a, b) for a in ["Bo'Bo'"] for b in ["a"]]
    # wagons = []
    # rs = RollingStock(locs, wagons)
    # train  = rs.get_train(12)
    # for _ in xrange(100):
    #     train2 = rs.get_neighbor_train(train, fixed_length_trains=True)
    #     plt.plot(train2.loadvector-train.loadvector)

    # plt.show(block=True)
