"""
Unit tests for chronometer.py
"""

import os
import numpy as np
import unittest
import gibbs_chronometer as gc
import pandas as pd
from isochrones import StarModel
from isochrones.mist import MIST_Isochrone
import priors


class ChronometerTestCase(unittest.TestCase):
    """
    Tests for chronometer.py
    """

    def test_data_file_validity(self):
        """
        Test that all the values in the input files are within reasonable
        ranges.
        age,age_err,bv,bv_err,period,period_err,v,v_err,ks,ks_err,vk,vk_err,
        j,j_err,h,h_err,k,k_err,parallax,parallax_err,Teff,Teff_err,logg,
        logg_err,feh,feh_err,mass,mass_err,Av,Av_err
        """
        bounds = [[0, 13.7], [0, 10], [0, 200], [0, 100], [-10, 10], [0, 10],
                  [0, 30], [0, 10], [0, 30], [0, 10], [0, 30], [0, 10],
                  [0, 30], [0, 10], [0, 30], [0, 10], [0, 30], [0, 10],
                  [0, 1e5], [0, 100], [1000, 10000], [1, 1000], [2, 6],
                  [0, 3], [-4, 6], [0, 10], [0, 100], [0, 3], [-1, 10],
                  [0, 5]]

        b = True
        for i, bound in enumerate(bounds):
            b += (bound[0] < \
                  d.iloc[:, i][np.isfinite(d.iloc[:, i])]) & \
                (d.iloc[:, i][np.isfinite(d.iloc[:, i])] <
                 bound[1])
        self.assertTrue(b.all())

    def test_lnprob_returns_finite(self):
        """
        Make sure lnprob returns finite.
        """
        args = [mods, d.period.values, d.period_err.values,
                d.bv.values, d.bv_err.values, "both"]
        self.assertTrue(np.isfinite((gc.lnprob(p0, *args))))

    def test_iso_single_lnlike_returns_finite(self):
        """
        Make sure iso_single_lnlike_returns_finite.
        """
        self.assertTrue(np.isfinite((gc.iso_single_lnlike(one_star_pars,
                                                          mods[0]))))

    def test_lnprior_returns_finite(self):
        """
        Make sure lnprior returns a finite number.
        """
        self.assertTrue(np.isfinite(gc.lnprior(p0)))

    def test_parameter_assignment_reasonable(self):
        Nstars = 3
        N, ln_mass, ln_age, feh, ln_distance, Av = \
        gc.parameter_assignment(p0[3:], "iso")
        self.assertTrue(N == len(ln_mass))
        self.assertTrue(type(N) == int)
        self.assertTrue(Nstars == N)

        N, ln_mass, ln_age, feh, ln_distance, Av = \
            gc.parameter_assignment(p0, "both")
        self.assertTrue(N == len(ln_mass))
        self.assertTrue(type(N) == int)
        self.assertTrue(Nstars == N)

        N, ln_mass, ln_age, feh, ln_distance, Av = \
            gc.parameter_assignment(one_star_pars, "both")
        self.assertTrue(type(N) == int)

    def test_transform_parameters_correct(self):
        Nstars = 3
        p, N = gc.transform_parameters(p0[3:], "iso", False)
        self.assertTrue(len(p) == Nstars*5)
        self.assertTrue(Nstars == N)
        p, N = gc.transform_parameters(p0, "iso", True)
        self.assertTrue(len(p == Nstars*5))
        self.assertTrue(Nstars == N)
        pars, lnages = gc.transform_parameters(p0[:6], "gyro", False)
        self.assertTrue(len(pars) == 3)
        self.assertTrue(len(lnages) == Nstars)
        pars, lnages = gc.transform_parameters(p0, "gyro", True)
        self.assertTrue(len(pars) == 3)
        self.assertTrue(len(lnages) == Nstars)

    def test_iso_lnprior_one_star(self):
        self.assertTrue(np.isfinite(gc.iso_lnprior(np.array(one_star_pars))))

if __name__ == "__main__":

    # Global variables.
    DATA_DIR = "/Users/ruthangus/projects/chronometer/chronometer/data"
    d = pd.read_csv(os.path.join(DATA_DIR, "data_file.csv"))
    p0, mods = gc.pars_and_mods(DATA_DIR)
    one_star_pars = [np.log(1.), np.log(4.), 0., np.log(1000.), 0.]

    unittest.main()
