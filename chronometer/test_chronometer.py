"""
Unit tests for chronometer.py
"""

import os
import numpy as np
import unittest
from chronometer import lnprior
import pandas as pd


class ChronometerTestCase(unittest.TestCase):
    """
    Tests for chronometer.py
    """

    def test_lnprior_returns_finite(self):
        """
        Make sure lnprior returns a finite number.
        """
        gc = np.array([.7725, .601, .5189])
        ages = np.array([4.56, .5])
        masses = np.array([1., 1.])
        fehs = np.array([0., 0.])
        ds = np.array([10., 10.])
        Avs = np.array([0., 0.])
        p0 = np.concatenate((gc, np.log(masses), np.log(ages), fehs,
                             np.log(ds), Avs))

        b = np.isfinite(lnprior(p0))
        self.assertTrue(b)

    def test_data_file_validity(self):
        DATA_DIR = "/Users/ruthangus/projects/chronometer/chronometer/data"
        d = pd.read_csv(os.path.join(DATA_DIR, "data_file.csv"))
        bounds = [[0, 13.7], [0, 10], [0, 200], [0, 100], [-10, 10], [0, 10],
                  [0, 30], [0, 10], [0, 30], [0, 10], [0, 30], [0, 10],
                  [0, 30], [0, 10], [0, 30], [0, 10], [0, 1e5], [0, 100],
                  [1000, 10000], [1, 1000], [2, 6], [0, 3], [-4, 6], [0, 10]]

        print(np.shape(d))
        print(len(bounds))
        assert 0
        b = True
        for i, bound in enumerate(bounds):
            b += (bound[0] < d[:, i][np.isfinite(d[:, i])]) & \
                (d[:, i][np.isfinite(d[:, i])] < bound[1])

        # b = (0 < d.age[np.isfinite(d.age)]) & \
        #     (d.age[np.isfinite(d.age)] < 13.7) \
        #     & (0 < d.age_err[np.isfinite(d.age_err)]) & \
        #     (d.age_err[np.isfinite(d.age_err)] < 10) & \
        #     (0 < d.period[np.isfinite(period)]) & \
        #     (d.period[np.isfinite(period)] < 200) & \
        #     (0 < d.period_err[np.isfinite(period_err)]) & \
        #     (d.period_err[np.isfinite(period_err)] < 100) & \
        #     (-10 < d.bv[np.isfinite(d.bv)]) & \
        #     (d.bv[np.isfinite(d.bv)] < 10) & \
        #     (0 < d.bv_err[np.isfinite(d.bv_err)]) & \
        #     (d.bv_err[np.isfinite(d.bv_err)] < 10) & \
        #     # (0 < d.v) & (d.v < 30) & (0 < d.v_err) & (d.v_err < 10) & \
        #     # (0 < d.ks) & (d.ks < 30) & (0 < d.ks_err) & (d.ks_err < 10) & \
        #     # (0 < d.vk) & (d.vk < 30) & (0 < d.vk_err) & (d.vk_err < 10) & \
        #     # (0 < d.j) & (d.j < 30) & (0 < d.j_err) & (d.j_err < 10) & \
        #     # (0 < d.h) & (d.h < 30) & (0 < d.h_err) & (d.h_err < 10) & \
        #     # (0 < d.k) & (d.k < 30) & (0 < d.k_err) & (d.k_err < 10) & \
        #     # (0 < d.parallax) & (d.parallax < 1e5) & (0 < d.parallax_err) & \
        #     # (d.parallax_err < 100) &\
        #     # (1000 < d.Teff) & (d.Teff < 10000) & (1 < d.Teff_err) & \
        #     # (d.Teff_err < 1000) & \
        #     # (2. < d.logg) & (d.logg < 6) & (0 < d.logg_err) & \
        #     # (d.logg_err < 3) &\
        #     # (-4. < d.feh) &(d.feh < 6) & (0 < d.feh_err) & (d.feh_err < 10)

        # self.assertTrue(b.all())


if __name__ == "__main__":
    unittest.main()
