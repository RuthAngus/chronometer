"""
Unit tests for chronometer.py
"""

import os
import numpy as np
import unittest
from chronometer import lnprior, lnprob, replace_nans_with_inits, \
    make_param_dict
import pandas as pd
from isochrones import StarModel
from isochrones.mist import MIST_Isochrone


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
        """
        Test that all the values in the input files are within reasonable
        ranges.
        age,age_err,bv,bv_err,period,period_err,v,v_err,ks,ks_err,vk,vk_err,
        j,j_err,h,h_err,k,k_err,parallax,parallax_err,Teff,Teff_err,logg,
        logg_err,feh,feh_err,mass,mass_err,Av,Av_err
        """
        DATA_DIR = "/Users/ruthangus/projects/chronometer/chronometer/data"
        d = pd.read_csv(os.path.join(DATA_DIR, "data_file.csv"))
        bounds = [[0, 13.7], [0, 10], [0, 200], [0, 100], [-10, 10], [0, 10],
                  [0, 30], [0, 10], [0, 30], [0, 10], [0, 30], [0, 10],
                  [0, 30], [0, 10], [0, 30], [0, 10], [0, 30], [0, 10],
                  [0, 1e5], [0, 100], [1000, 10000], [1, 1000], [2, 6],
                  [0, 3], [-4, 6], [0, 10], [0, 100], [0, 3], [-1, 10],
                  [0, 5]]

        b = True
        for i, bound in enumerate(bounds):
            b += (bound[0] < d.iloc[:, i][np.isfinite(d.iloc[:, i])]) & \
                (d.iloc[:, i][np.isfinite(d.iloc[:, i])] < bound[1])
        self.assertTrue(b.all())

    def test_lnprobs(self):
        DATA_DIR = "/Users/ruthangus/projects/chronometer/chronometer/data"
        # The parameters
        gc = np.array([.7725, .601, .5189])
        d = pd.read_csv(os.path.join(DATA_DIR, "data_file.csv"))

        d = replace_nans_with_inits(d)
        p0 = np.concatenate((gc, np.log(d.mass.values),
                            np.log(d.age.values), d.feh.values,
                            np.log(1./d.parallax.values*1e3),
                            d.Av.values))
        params_init = p0*1
        # iso_lnlike preamble.
        mist = MIST_Isochrone()
        mods = []
        for i in range(len(d.period.values)):
            param_dict = make_param_dict(d, i)
            # Remove missing parameters from the dict.
            param_dict = {k: param_dict[k] for k in param_dict if
                        np.isfinite(param_dict[k]).all()}
            mods.append(StarModel(mist, **param_dict))
        print("iso")
        p0 = p0[3:]
        args = [mods, "iso"]
        truths = [np.log(1), None, np.log(4.56), np.log(2.5), 0., None,
                np.log(10), np.log(2400), 0., None]
        labels = ["$\ln(Mass_1)$", "$\ln(Mass_2)$", "$\ln(Age_1)$",
                  "$\ln(Age_2)$", "$[Fe/H]_1$", "$[Fe/H]_2$", "$\ln(D_1)$",
                  "$\ln(D_2)$", "$A_{v1}$", "$A_{v2}$"]
        self.assertTrue(np.isfinite((lnprob(p0, *args))))


if __name__ == "__main__":
    unittest.main()
