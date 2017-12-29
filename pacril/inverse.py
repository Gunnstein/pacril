# -*- coding: utf-8 -*-
import numpy as np
import scipy.signal
import scipy.linalg
import scipy.optimize

__all__ = ['find_influenceline_lstsq', 'find_influenceline_fourier',
           'find_loadmagnitude_vector']


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


def find_influenceline_lstsq(z, f):
    """Find influenceline from response and load vector by lstsq method.

    By deconvolving the measured response `z` and the load vector `f` the
    influenceline vector `l` can be extracted.

    For more informations see
        G. T. Froeseth, A. Ronnquist, D. Cantero, O. Oiseth. Influenceline
        extraction by deconvolution in the frequency domain. Computer and
        Structures 189:21-30 (2017)

    Arguments
    ---------
    z : ndarray
        Response vector.

    f : ndarray
        Load vector.

    Returns
    -------
    ndarray
        The estimated influence line
    """
    Nz = z.size
    Nf = f.size
    f1 = np.zeros_like(z)
    f1[:Nf] = f
    Nl = Nz - Nf + 1
    F = scipy.linalg.toeplitz(f1, np.zeros(Nl))
    l, _, _, _ = np.linalg.lstsq(F, z)
    return l


def find_influenceline_fourier(z, f, alpha=0.):
    """Find influenceline from response and load vector by FD method.

    By deconvolving the measured response `z` and the load vector `f` the
    influenceline vector `l` can be extracted. The Fourier domain method with
    regularization function M(omega) = 1.0 is used in this function.

    For more informations see
        G. T. Froeseth, A. Ronnquist, D. Cantero, O. Oiseth. Influenceline
        extraction by deconvolution in the frequency domain. Computer and
        Structures 189:21-30 (2017)

    Arguments
    ---------
    z : ndarray
        Response vector.

    f : ndarray
        Load vector.

    alpha : float
        Regularization parameter for the fourier method.

    Returns
    -------
    ndarray
        The estimated influence line
    """
    Nz = z.size
    Nf = f.size
    Nl = Nz - Nf + 1
    Z = np.fft.rfft(z, Nz)
    F = np.fft.rfft(f, Nz)
    L = Z * np.conjugate(F) / (np.abs(F)**2 + alpha)
    l = np.fft.irfft(L)[:Nl]
    return l


def find_loadmagnitude_vector(z, l, xp, fx=10.):
    """Determine the load magnitude vector by Moses algorithm

    Finds the load magnitude vector `p` from responsevector `z`, influenceline
    `l` and the load position vector `xp` by Moses original algorithm.

    Arguments
    ---------
    z : ndarray
        Response vector
    """
    nxp = np.round(xp*fx).astype(np.int)[1:-1]
    Nz = z.size
    Nl = l.size
    Nf = Nz - Nl + 1
    l1 = np.zeros_like(z)
    l1[:Nl] = l
    IL = scipy.linalg.toeplitz(l1, np.zeros(Nf))[:, nxp]
    p, _, _, _ = np.linalg.lstsq(IL, z)
    return p
