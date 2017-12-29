# -*- coding: utf-8 -*-
import numpy as np
import scipy

__all__ = ['find_influenceline_lstsq', 'find_influenceline_fourier',
           'find_loadmagnitude_vector']


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
