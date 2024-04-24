# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import numpy as np
import scipy.linalg
import scipy.sparse


__all__ = ['find_influenceline_lstsq', 'find_influenceline_fourier',
           'find_loadmagnitude_vector', 'find_influenceline_fastmatrix',
           'find_lag_phase_method']


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
    l, _, _, _ = scipy.linalg.lstsq(F, z)
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


def find_influenceline_fastmatrix(z, f):
    """Find influenceline from response and load vector by fast lstsq method.

    By deconvolving the measured response `z` and the load vector `f` the
    influenceline vector `l` can be extracted. The sparseness of the load
    vector is exploited with the Levinson-Durbin algorithm to solve the
    system in O(Nl^2) time instead of O(NzNl^2) for the conventional
    matrix method.

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
    fpad = np.pad(f, (0, z.size - f.size), mode='constant',
                  constant_values=(0, 0))
    F = scipy.linalg.toeplitz(fpad, np.zeros(z.size-f.size+1, dtype=np.double))
    Fcsr = scipy.sparse.csr_matrix(F)
    FTF = Fcsr.transpose().dot(Fcsr)
    FTz = Fcsr.transpose().dot(z)
    return scipy.linalg.solve_toeplitz(
        (FTF.getcol(0).toarray(), FTF.getrow(0).toarray()), FTz)


def find_loadmagnitude_vector(z, l, xp, fx=10.):
    """Determine the load magnitude vector by Moses algorithm

    Finds the load magnitude vector `p` from responsevector `z`, influenceline
    `l` and the load position vector `xp` by Moses original algorithm.

    Arguments
    ---------
    z : ndarray
        Response vector
    """
    nxp = np.round(xp*fx).astype(int)[1:-1]
    Nz = z.size
    Nl = l.size
    Nf = Nz - Nl + 1
    l1 = np.zeros_like(z)
    l1[:Nl] = l
    IL = scipy.linalg.toeplitz(l1, np.zeros(Nf))[:, nxp]

    p, _, _, _ = scipy.linalg.lstsq(IL, z)
    return p


def find_lag_phase_method(y1, y2):
    """Determines the lag between two signals by correlation

    The function returns the number of sample points necessary to achieve
    maximum correlation between signal y1 and y2.

    Arguments
    ---------
    y1, y2 : ndarray
        Signals to find the lag between.

    Returns
    -------
    int
        The number of sample points to skew y1 to achieve maximum correlation
        with y2
    """
    corr = np.correlate(y1, y2, mode='full')
    nmax = np.argmax(corr)
    return int(round(y2.size - nmax)) - 1

def find_speed_phase_method(y1, y2, dx, sampling_rate):
    """Determines the speed by phase correlation method

    The function uses the phase correlation to determine the speed

   Arguments
    ---------
    y1, y2 : ndarray
        Signals to find the lag between.
    dx : float
        Spatial separation along load path between sensors y1 and y2.
    sampling_rate : float
        Temporal sampling rate of signal (in Samples pr second).

    Returns
    -------
    float
        Estimated speed
    """
    nlag = find_lag_phase_method(y1, y2)
    Dt = float(nlag) / float(sampling_rate)
    return dx / Dt
