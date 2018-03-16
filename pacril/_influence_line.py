# -*- coding: utf-8 -*-
import numpy as np

__all__ = ['get_il_simply_supported_beam']

def get_il_simply_supported_beam(L, xi, fx=10.):
    """Returns the influence line for moment of a simply supported beam.

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
    return -l



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.plot(get_il_simply_supported_beam(10., .5))
    plt.show(block=True)