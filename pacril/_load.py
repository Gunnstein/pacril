# -*- coding: utf-8 -*-
import numpy as np
import scipy


__all__ = ['get_loadvector', 'join_loads', 'find_daf_EC3',
           'get_geometry_twoaxle_wagon', 'get_loadvector_twoaxle_wagon',
           'get_geometry_bogie_wagon', 'get_loadvector_bogie_wagon',
           'get_geometry_jacobs_wagon', 'get_loadvector_jacobs_wagon', ]


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


