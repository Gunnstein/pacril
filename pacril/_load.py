# -*- coding: utf-8 -*-
import numpy as np
import scipy
import copy


__all__ = ['get_loadvector', 'join_loads', 'find_daf_EC3',
           'get_geometry_twoaxle_wagon', 'get_loadvector_twoaxle_wagon',
           'get_geometry_bogie_wagon', 'get_loadvector_bogie_wagon',
           'get_geometry_jacobs_wagon', 'get_loadvector_jacobs_wagon',
           'Load', 'Locomotive', 'TwoAxleWagon', 'BogieWagon',
           'JacobsWagon', 'Train', 'RollingStock', ]


def get_loadvector(p, xp, fx=10.):
    """Define load vector from loads p and geometry vector xp.

    The vehicle is defined by loads p and geometry xp, see figure below.

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

    fx : Optional[float]
        The number of samples per unit coordinate.

    Returns
    -------
    ndarray
        The load vector.
    """
    nxp = np.round(fx*np.asfarray(xp)).astype(np.int)
    f = np.zeros(nxp.max()+1, dtype=np.float)
    f[nxp[1:-1]] = p
    return f


def get_geometry_twoaxle_wagon(a, b):
    """Define geometry vector xp for a twoaxle wagon.

    The two axle wagon is defined geometry parameter a and b, see figure below.

            Type:          Twoaxle wagon
                           +------------+
            Axleload:      |            |
                        (--+------------+--)
                             O        O
            Geometry:   |----|--------|----|
                          a       b     a

    Arguments
    ---------
    a,b : float
        Defines the geometry of the vehicle, i.e the start, end and axle
        positions.

    Returns
    -------
    ndarray
        Geometry vector xp.
    """
    return np.array([0., a, a+b, 2.*a+b])


def get_loadvector_twoaxle_wagon(p, a, b, fx=10.):
    """Define load vector for a twoaxle wagon.

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
    p : float or ndarray
        Defines the axle loads, if p is a float it assumes equal axle load
        on each axle, if p is a ndarray it defines each axle load
        individually. Note that the array must be of size `x.size-2` or
        an exception will be raised.

    a,b : float
        Defines the geometry of the vehicle, i.e the start, end and axle
        positions.

    fx : Optional[float]
        The number of samples per unit coordinate, defines the coordinate
        of the load.

    Returns
    -------
    ndarray
        The load vector
    """
    xp = get_geometry_twoaxle_wagon(a, b)
    return get_loadvector(p, xp, fx)


def get_geometry_bogie_wagon(a, b, c):
    """Define geometry vector xp for a bogie wagon.

    Geometry of a bogie wagon is defined by parameters a, b and c, see figure
    below.

            Type:               Bogie wagon
                           +-------------------+
            Axleload:      |                   |
                        (--+-------------------+--)
                             O   O       O   O
                        |------|-----------|------|
            Geometry:      a         b         a
                             |---|       |---|
                               c           c


    Arguments
    ---------
    a,b,c : float
        Defines the geometry of the vehicle, i.e the start, end and axle
        positions.

    Returns
    -------
    ndarray
        Geometry vector xp.
    """
    return np.cumsum([0., a-c/2., c, b-c, c, a/2.])


def get_loadvector_bogie_wagon(p, a, b, c, fx=10.):
    """Define a load vector for a bogie wagon.

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
    p : float or ndarray
        Defines the axle loads, if p is a float it assumes equal axle load
        on each axle, if p is a ndarray it defines each axle load
        individually. Note that the array must be of size `x.size-2` or
        an exception will be raised.

    a,b,c : float
        Defines the geometry of the vehicle, i.e the start, end and axle
        positions.

    fx : Optional[float]
        The number of samples per unit coordinate, defines the coordinate
        of the load.

    Returns
    -------
    ndarray
        The load vector.
    """
    xp = get_geometry_bogie_wagon(a, b, c)
    return get_loadvector(p, xp, fx)


def get_geometry_jacobs_wagon(a, b, c):
    """Define geometry vector xp for a jacobs wagon.

    Geometry of a jacobs wagon is defined by parameters a, b and c, see figure
    below.

            Type:                     Jacobsbogie wagon
                           +----------------------------------+
            Axleload:      |                                  |
                        (--+----------------)(----------------+--)
                             O   O        O    O        O   O
                        |------|------------|-------------|------|
            Geometry:      a         b             b          a
                             |---|        |----|        |---|
                               c            c             c

    Arguments
    ---------
    a,b,c : float
        Defines the geometry of the vehicle, i.e the start, end and axle
        positions.

    Returns
    -------
    ndarray
        Geometry vector
    """
    return np.cumsum([0., a-c/2., c, b-c, c, b-c, c, a-c/2.])


def get_loadvector_jacobs_wagon(p, a, b, c, fx=10):
    """Define a load vector for a jacobsbogie wagon.

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
    p : float or ndarray
        Defines the axle loads, if p is a float it assumes equal axle load
        on each axle, if p is a ndarray it defines each axle load
        individually. Note that the array must be of size `x.size-2` or
        an exception will be raised.

    a,b,c : float
        Defines the geometry of the vehicle, i.e the start, end and axle
        positions.

    fx : Optional[float]
        The number of samples per unit coordinate, defines the coordinate
        of the load.

    Returns
    -------
    ndarray
        The load vector.
    """
    xp = get_geometry_jacobs_wagon(a, b, c)
    return get_loadvector(p, xp, fx)


def join_loads(*args):
    """Join a sequence of loads.

    Assume `f0`, `f1` and `f2` are load vectors, then

        f = join_loads(f0, f1, f2)

    yields a new load vector where f0-f2 are joined in sequence.

    Arguments
    ---------
    f0, f1, f2,...,fn : ndarray
        Load vectors to be joined

    Returns
    -------
    ndarray
        The joined load vectors
    """
    return np.concatenate(args)


def find_daf_EC3(v, L):
    """Dynamic amplification factor according to EC1-2 annex D

    Arguments
    ---------
    v : float
        Speed in km / h, should be smaller than 200 km/h.
    L : float
        Determinant length in meters.
    n0 : float
        The first natural bending frequency of the bridge loaded by permanent
        actions.

    Returns
    -------
    float
    """
    if np.any(v > 200):
        raise ValueError("Speed must be smaller than 200 km/h")
    vms = v / 3.6
    if L <= 20:
        K = vms / 160.
    else:
        K = vms / (47.16*L**0.408)
    phi1 = K / ( 1 - K + K**4)
    phi11 = 0.56 * np.exp(-L**2/100.)
    return 1 + .5*(phi1+.5*phi11)


class BaseLoad(object):
    """Parent object of load objects, all other loadobjects are children of
    this class. All children should define xp and p upon initialization.

    """
    def __init__(self):
        self.fx = 10.

    def copy(self):
        return copy.deepcopy(self)

    @property
    def loadvector(self):
        return get_loadvector(self.p, self.xp, self.fx)


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
        nxp = np.round(self.fx*np.asfarray(self.xp)).astype(np.int)
        Nl = influence_line.size
        Nf = nxp.max()+1
        Nz = Nl+Nf-1
        z = np.zeros(Nz, dtype=np.float)
        for ni, pi in zip(nxp[1:-1], self.p):
            z[ni:ni+Nl] += pi * influence_line
        return z

    @property
    def nloads(self):
        return len(self.xp)-2

    def __getattribute__(self, name):
        value = super(BaseLoad, self).__getattribute__(name)
        if (name == "p"):
            if isinstance(value, float) or isinstance(value, np.float):
                return np.array([value] * self.nloads)
            else:
                return np.array(value)
        else:
            return value


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


class BaseVehicle(BaseLoad):
    """Parent object of vehicles, extends the definition of load to contain
    a vehicle properties and methods such as definition of axles and goods.
    Children must define xp, p and pempty.
    """
    @property
    def naxles(self):
        return len(self.xp)-2

    def __getattribute__(self, name):
        value = super(BaseVehicle, self).__getattribute__(name)
        if (name == "pempty"):
            if isinstance(value, float) or isinstance(value, np.float):
                return np.array([value] * self.nloads)
            else:
                return np.array(value)
        else:
            return value

    @property
    def goods_transported(self):
        p = self.p
        pempty = self.pempty
        return (p - pempty).sum()


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

    @property
    def xp(self):
        loc = self.locomotive
        wagons = self.wagons
        xp0 = [loc.xp] + [wag.xp for wag in wagons]
        xpstart = np.cumsum([0.] + [x[-1] for x in xp0])
        xp = np.array(
            [0.] + [xpi0 + xpij for xpi0, xpi in zip(xpstart[:-1], xp0)
                    for xpij in xpi[1:-1]] + [xpstart[-1]]
            )
        return xp

    @property
    def p(self):
        loc = self.locomotive
        wagons = self.wagons
        p = np.concatenate([loc.p] + [wag.p for wag in wagons])
        return p

    @property
    def pempty(self):
        loc = self.locomotive
        wagons = self.wagons
        p = np.concatenate([loc.pempty] + [wag.pempty for wag in wagons])
        return p

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

    def remove_wagon(self, ix):
        """Remove wagon from train

        Arguments
        ---------
        ix : int
            The index of the wagon to remove
        """
        self.wagons.pop(ix)


class RollingStock(object):
    """Rolling stock contains the possible sets of locomotives and wagons.

    A rolling stock object contains locomotives and wagons and methods to
    generate and assemble trains.


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
        Nwag_min,Nwag_max : int
            The minimum and maximum number of wagons that the train can consist
            of.
        """
        Nwag = train.nwagons
        train_new = Train(train.locomotive, list(train.wagons))
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
        elif n == -2:
            n = np.random.randint(0, Nwag)
            train_new.insert_wagon(n, self.choose_wagons(1)[0])
        elif n == -1:
            train_new.swap_locomotive(self.choose_locomotive())
        else:
                train_new.swap_wagon(n, self.choose_wagons(1)[0])
        return train_new


if __name__ == '__main__':


    import matplotlib.pyplot as plt
    import _influence_line
    import timeit
    xploc = np.array([0., 2.2, 5.4, 9.5, 12.7, 14.9])
    ploc = np.array([18., 18., 18., 18.])
    loc = Locomotive(xploc, ploc)

    wagons = [TwoAxleWagon(p, 3., 4., 3.) for p in [12., 4.]]
    rs = RollingStock([loc], wagons)

    x0 = rs.get_train(11)
    l = _influence_line.get_il_simply_supported_beam(4., .2)
    plt.plot(x0.apply(l))

    x1 = rs.get_neighbor_train(x0)
    plt.plot(x1.apply(l))
    plt.plot(x0.apply(l))
    plt.show(block=True)




