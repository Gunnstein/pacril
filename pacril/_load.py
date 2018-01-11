# -*- coding: utf-8 -*-
import numpy as np
import scipy


__all__ = ['get_loadvector', 'get_twoaxle_wagon',
           'get_bogie_wagon', 'get_jacobs_wagon']



def get_loadvector(p, xp, fx=10.):
    """Define a load vector from loads p and geometry vector xp.

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


def get_twoaxle_wagon(p, a, b, fx=10):
    """Define a load vector for a twoaxle wagon.

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
    """
    xp = np.array([0., a, a+b, 2.*a+b])
    return get_loadvector(p, xp, fx)


def get_bogie_wagon(p, a, b, c, fx=10.):
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
    """
    xp = np.cumsum([0., a-c/2., c, b-c, c, a/2.])
    return get_loadvector(p, xp, fx)


def get_jacobs_wagon(p, a, b, c, fx=10):
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
    """
    xp = np.cumsum([0., a-c/2., c, b-c, c, b-c, c, a-c/2.])
    return get_loadvector(p, xp, fx)


def join_loads(*args):
    """Join a sequence of loads.

    Assume `f0`, `f1` and `f2` are load vectors, then

        f = join_loads(f0, f1, f2)

    yields a new load vector where f0-f2 are joined in sequence.
    """
    return np.concatenate(args)
