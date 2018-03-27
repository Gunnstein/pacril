# -*- coding: utf-8 -*-
import numpy as np

__all__ = ['get_il_simply_supported_beam',
           'get_il_two_span_simply_supported_beam']


def get_il_simply_supported_beam(L, xi, fx=10.):
    """Returns the influence line for moment of simply supported beam.

    The function returns the influence line for bending moment measured at
    location `xi*L` from the leftmost support of a simply supported beam with
    length `L`. See figure below.

                    1
                    |
                    |
                    v
        o-----------------o
       / \               / \
      +++++             =====
        |-----|-----------|--------> x
        0    xi*L         L
              ^
              |
        Sensor location

    Arguments
    ---------
    L : float
        Length of the simply supported beam
    xi : float
        Relative distance of whole length L from the leftmost support to the
        sensor location. Must be in [0., 1.] or a ValueError will be raised.
    fx : Optional[float]
        The sampling frequency per length unit.

    Returns
    -------
    ndarray
        Influence line

    Raises
    ------
    ValueError
        If `xi` is not in range [0., 1.]
    """
    if (xi < 0) | (xi > 1.):
        raise ValueError(
            "`xi` must be in [0., 1.]")
    dx = 1. / float(fx)
    x = np.arange(0., np.round(L*fx) + 1) * dx
    l = np.zeros_like(x)
    Nl = np.round(xi * x.size).astype(np.int)
    l[:Nl] = (xi-1.)*x[:Nl]/L
    l[Nl:] = (x[Nl:] / L - 1.) * xi
    return -l * L


def get_il_two_span_simply_supported_beam(L, xi, fx=10.):
    """Return the influence line for moment of two-span simply supported beam.

    The function returns the influence line for bending moment measured at
    location `xi*L` from the leftmost support of a simply supported beam with
    length `L`. See figure below.
                    1
                    |
                    |
                    v
        o-----------------o-----------------o
       / \               / \               / \
      +++++             =====             =====
        |-----|-----------|-----------------|--------> x
        0    xi*L        L/2                L
              ^
              |
        Sensor location

    Arguments
    ---------
    L : float
        Length of the two span simply supported beam
    xi : float
        Relative distance of whole length L from the leftmost support to the
        sensor location. Must be in [0., 1.] or a ValueError will be raised.
    fx : Optional[float]
        The sampling frequency per length unit.

    Returns
    -------
    ndarray
        Influence line

    Raises
    ------
    ValueError
        If `xi` is not in range [0., 1.]
    """
    if (xi < 0) | (xi > 1.):
        raise ValueError(
            "`xi` must be in [0., 1.]")
    dx = 1. / float(fx)
    x = np.arange(0., np.round(L*fx) + 1) * dx
    l = np.zeros_like(x)
    xil = xi
    for j, xp in enumerate(x):
        xip = xp / L
        if 0. <= xip <= .5:
            Az = 2.0*xip**3 - 2.5*xip + 1.0
            Bz = 3.0*xip - 4.0*xip**3
            Cz = 2.0*xip**3 - 0.5*xip
        else:
            Az = 2.0*(1.0-xip)**3 + 0.5*xip - 0.5
            Bz = 3.0*(1.0-xip) - 4.0*(1.0-xip)**3
            Cz = 2.0*(1.0-xip)**3 + 2.5*xip - 1.5

        if xil <= .5:
            if xil <= xip:
                l[j] = Az*xil*L
            else:
                l[j] = Az*xil*L - (xil-xip)*L
        else:
            if xil <= xip:
                l[j] = Az*xil*L + Bz*(xil-0.5)*L
            else:
                l[j] = Az*xil*L + Bz*(xil-0.5)*L - (xil-xip)*L
    return l


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.figure()
    for xil in [.25, .5, .75]:
        plt.plot(get_il_two_span_simply_supported_beam(8., xil),
                 label="xi={0:.2f}".format(xil))
    plt.plot(get_il_simply_supported_beam(4., .5), label='one-span')
    plt.ylabel('[Nm / N]')
    plt.legend()
    plt.show(block=True)