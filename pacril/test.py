# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import numpy as np
import scipy
import copy
import json
from . import *
import unittest
from .data import UICTrainFactory, ECTrainFactory
from .data import influence_lines

class TestPacril(unittest.TestCase):
    def test_get_coordinate_vector(self):
        est = get_coordinate_vector([0.]*11, 10)
        tru = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        np.testing.assert_almost_equal(est, tru, 2)

    def test_get_loadvector(self):
        f_est = get_loadvector(10., np.array([0., 1.0, 3.0, 5.0]), 10)
        f_true = np.array([
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,10.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_almost_equal(f_est, f_true, 2)

    def test_get_loadvector_twoaxle_wagon(self):
        f_est = get_loadvector_twoaxle_wagon([10., 11], 1.6, 9., 10)
        f_true = np.array([
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0,11.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_almost_equal(f_est, f_true, 2)

    def test_get_loadvector_bogie_wagon(self):
        f_est = get_loadvector_bogie_wagon([10., 11., 12., 13],
                                           2.5, 9.9, 1.8, 10)
        f_true = np.array([
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 11.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 12.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 13.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_almost_equal(f_est, f_true, 2)

    def test_get_loadvector_jacobs_wagon(self):
        f_est = get_loadvector_jacobs_wagon(
            [10., 11., 12., 13., 14., 15.], 2.5, 15., 1.8, 10.)
        f_true = np.array([
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,11.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,12.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0,13.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,14.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,15.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_almost_equal(f_est, f_true, 2)

    def test_find_influenceline_lstsq(self):
        l_true = scipy.bartlett(5*10)
        f = get_loadvector_jacobs_wagon(
            [10., 20., 13., 19., 21., 7], 2.5, 17., 1.8, 10)
        z = np.convolve(f, l_true)
        l_est = find_influenceline_lstsq(z, f)
        np.testing.assert_almost_equal(l_est, l_true)

    def test_find_influenceline_fourier(self):
        l_true = scipy.bartlett(5*10)
        f = get_loadvector_jacobs_wagon(
            [10., 20., 13., 19., 21., 7], 2.5, 17., 1.8, 10)
        z = np.convolve(f, l_true)
        l_est = find_influenceline_fourier(z, f)
        np.testing.assert_almost_equal(l_est, l_true)

    def test_find_loadmagnitude_vector(self):
        xp = np.array([0., 1.0, 4.3, 5.0])
        p_true = np.array([12., 15.])
        f = get_loadvector(p_true, xp, 10)
        l = scipy.bartlett(5*10)
        z = np.convolve(l, f)
        p_est = find_loadmagnitude_vector(z, l, xp, 10)
        np.testing.assert_almost_equal(p_est, p_true)

    def test_find_daf_EC1(self):
        v = np.asfarray([10., 33., 100., 120., 150., 175., 200.])
        est = find_daf_EC1(v, 7.)
        tru = np.array([1.09460162, 1.11615409, 1.19069436, 1.21703429,
                        1.26073599, 1.30133390, 1.34593202])
        np.testing.assert_almost_equal(est, tru, 4)
        est = find_daf_EC1(v, 25.)
        tru = np.array([1.00831790, 1.02784818, 1.0943087, 1.11742934,
                        1.15544947, 1.19047617,1.22875845])
        np.testing.assert_almost_equal(est, tru, 4)


class TestLoad(unittest.TestCase):
    def setUp(self):
        self.xptrue = np.array([0., 2.2, 5.4, 9.5, 12.7, 14.9])
        self.ptrue = np.array([18., 18., 18., 18.])
        self.load = Load(self.xptrue.copy(), self.ptrue.copy())
        self.nloads = 4
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
        self.influence_line = scipy.bartlett(5*10)
        self.response = np.convolve(self.loadvector, self.influence_line,
                                    'full')

    def test_xp(self):
        np.testing.assert_allclose(self.load.xp, self.xptrue)

    def test_p(self):
        np.testing.assert_allclose(self.load.p, self.ptrue)

    def test_nloads(self):
        np.testing.assert_equal(self.load.nloads, self.nloads)

    def test_loadvector(self):
        np.testing.assert_allclose(self.load.loadvector, self.loadvector)

    def test_apply(self):
        np.testing.assert_allclose(self.load.apply(self.influence_line),
                                   self.response)


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


class TestThreeAxleWagon(TestLocomotive):
    def setUp(self):
        self.load = ThreeAxleWagon(np.array([13., 21., 4.]), 2., 3.,
                                   np.array([1., 3., 1.]))
        self.xptrue = np.array([0.,  2.,  5.,  8., 10.])
        self.ptrue = np.array([13., 21., 4.])
        self.naxles = 3
        self.goods_transported = 13.+21.+4.-1.-3.-1.

        self.loadvector = np.array([
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 13., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 21., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 4., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0.])



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
        self.load = self.rs.get_random_train(15)

        self.traintrue = Train(locs[0], wags*15)
        self.xptrue = self.traintrue.xp
        self.ptrue = self.traintrue.p
        self.naxles = self.traintrue.naxles
        self.goods_transported = self.traintrue.goods_transported
        self.loadvector = self.traintrue.loadvector
        self.nwagontrue = self.traintrue.nwagons

    def test_get_neighbor_train(self):
        x0 = self.rs.get_random_train(25)
        for n in range(10000):
            self.rs.get_neighbor_train(x0)


class TestPacrilJSONDeEncoder(unittest.TestCase):
    def setUp(self):
        xploc = np.array([0., 2.2, 5.4, 9.5, 12.7, 14.9])
        ploc = np.array([18., 18., 18., 18.])
        loc = Locomotive(xploc, ploc)

        wagons = [TwoAxleWagon(p, a, b, 3.)
                  for p in [12., 4., 14., 9.]
                  for a, b in [(3., 4.), (5., 7.)]]
        self.rs = RollingStock([loc], wagons)
        self.JSONEncoder = serialize.PacrilJSONEncoder
        self.JSONDecoder = serialize.PacrilJSONDecoder

    def test_load(self):
        for n in np.random.randint(10, 51, 500):
            tr_true = self.rs.get_random_train(n)
            s = json.dumps(tr_true, cls=self.JSONEncoder)
            tr_est = json.loads(s, cls=self.JSONDecoder)
            np.testing.assert_almost_equal(tr_true.loadvector,
                                           tr_est.loadvector)


class TestDataNorwegianPacrilJSONDeEncoder(TestPacrilJSONDeEncoder):
    def setUp(self):
        self.rs = data.NorwegianRollingStock(3, "f")
        self.JSONEncoder = serialize.PacrilJSONEncoder
        self.JSONDecoder = data.norway.PacrilJSONDecoder


class TestECFatigueTrains(unittest.TestCase):
    def setUp(self):
        self.trains = [ECTrainFactory(n) for n in range(1, 13)]
        self.weights = np.array([663.0, 530.0, 940.0, 510.0, 2160.0, 1431.0,
                                 1035.0, 1035.0, 296.0, 360.0, 1135.0, 1135.0])
        self.speeds = np.array([200.0, 160.0, 250.0, 250.0, 100.0, 100.0, 80.0,
                                100.0, 120.0, 120.0, 120.0, 100.0])
        self.lengths = np.array([262.10, 281.10, 385.52, 237.60,
                                 270.30, 333.10, 196.50, 212.50,
                                 134.80, 129.60, 198.50, 212.50])

    def test_weights(self):
        weights = np.array([T.p.sum() for T in self.trains])
        np.testing.assert_almost_equal(self.weights, weights)

    def test_speeds(self):
        speeds = np.array([T.locomotive.speed for T in self.trains])
        np.testing.assert_almost_equal(self.speeds, speeds)

    def test_lengths(self):
        lengths = np.array([T.xp.max() for T in self.trains])
        np.testing.assert_almost_equal(self.lengths, lengths)


class TestUICTrains(TestECFatigueTrains):
    def setUp(self):
        self.trains = [UICTrainFactory(n) for n in range(1, 22)]
        self.weights = np.array([126.5, 249.0, 166.0, 325.0, 205.1,
                                 358.9, 378.0, 281.5, 470.5, 685.5,
                                 294.0, 478.0, 732.0,  52.0, 346.0,
                                 406.0, 971.1, 260.0, 469.0, 944.0,
                                 2123.0])

        self.lengths = np.array([64.04, 77.18, 76.15, 96.98, 95.02,
                                 135.77, 117.99, 73.15, 173.16, 236.80,
                                 98.78, 171.14, 271.90, 46.50, 151.10,
                                 177.5, 262.24, 149.00, 231.70, 323.80,
                                 342.30])

    def test_speeds(self):
        for T in self.trains:
            assert T.locomotive.speed is None


class TestStandardInfluenceLines(unittest.TestCase):
    def test_get_standard_influence_line(self):
        f = influence_lines.get_standard_influence_line
        res = np.array([f(ilt, 13.0)[29] for ilt in range(1, 10)])
        true = np.array([1.0, 0.4461538461538462, 0.4461538461538462,
                         0.19913846153846154, 0.9087741547692307, 0.8783905322307692,
                         0.928142132923077, -0.2007662183846154, 0.9526960894615384])
        np.testing.assert_allclose(true, res)


if __name__ == '__main__':
    unittest.main()
