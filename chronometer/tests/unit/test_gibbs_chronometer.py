"""
Unit tests for gibbs_chronometer.py
"""

import os
import numpy as np
import gibbs_chronometer as gc
import pandas as pd
from isochrones import StarModel
from isochrones.mist import MIST_Isochrone
import priors


class TestClass:
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
        print("Test 1")
        bounds = [[0, 13.7], [0, 10], [0, 200], [0, 100], [-10, 10], [0, 10],
                  [0, 30], [0, 10], [0, 30], [0, 10], [0, 30], [0, 10],
                  [0, 30], [0, 10], [0, 30], [0, 10], [0, 30], [0, 10],
                  [0, 1e5], [0, 100], [1000, 10000], [1, 1000], [2, 6],
                  [0, 3], [-4, 6], [0, 10], [0, 100], [0, 3], [-1, 10],
                  [0, 5]]

        b = True
        DATA_DIR = "/Users/ruthangus/projects/chronometer/chronometer/data"
        d = pd.read_csv(os.path.join(DATA_DIR, "data_file.csv"))
        for i, bound in enumerate(bounds):
            b += (bound[0] < \
                  d.iloc[:, i][np.isfinite(d.iloc[:, i])]) & \
                (d.iloc[:, i][np.isfinite(d.iloc[:, i])] <
                 bound[1])
        assert b.all() == True

    def test_lnprob_returns_finite(self):
        """
        Make sure lnprob returns finite.
        """
        print("Test 2")
        DATA_DIR = "/Users/ruthangus/projects/chronometer/chronometer/data"
        d = pd.read_csv(os.path.join(DATA_DIR, "data_file.csv"))
        p0, mods = gc.pars_and_mods(DATA_DIR)
        args = [mods, d.period.values, d.period_err.values,
                d.bv.values, d.bv_err.values, "both"]
        assert (np.isfinite((gc.lnprob(p0, *args)))) == True

    def test_lnprior_returns_finite(self):
        """
        Make sure lnprior returns a finite number.
        """
        print("Test 3")
        DATA_DIR = "/Users/ruthangus/projects/chronometer/chronometer/data"
        p0, mods = gc.pars_and_mods(DATA_DIR)
        assert (np.isfinite(gc.lnprior(p0))) == True

    def test_parameter_assignment_reasonable(self):
        """
        Test the parameter assignment function.
        """
        print("Test 4")
        DATA_DIR = "/Users/ruthangus/projects/chronometer/chronometer/data"
        p0, mods = gc.pars_and_mods(DATA_DIR)
        one_star_pars = [np.log(1.), np.log(4.), 0., np.log(1000.), 0.]
        Nstars = 3
        N, ln_mass, ln_age, feh, ln_distance, Av = \
        gc.parameter_assignment(p0[3:], "iso")
        assert (N == len(ln_mass)) == True
        assert (type(N) == int) == True
        assert (Nstars == N) == True

        N, ln_mass, ln_age, feh, ln_distance, Av = \
            gc.parameter_assignment(p0, "both")
        assert (N == len(ln_mass)) == True
        assert (type(N) == int) == True
        assert (Nstars == N) == True

        N, ln_mass, ln_age, feh, ln_distance, Av = \
            gc.parameter_assignment(one_star_pars, "both")
        assert (type(N) == int) == True

    def test_transform_parameters_correct(self):
        """
        Test the transform parameters function.
        """
        DATA_DIR = "/Users/ruthangus/projects/chronometer/chronometer/data"
        p0, mods = gc.pars_and_mods(DATA_DIR)
        print("Test 5")
        Nstars = 3
        p, N = gc.transform_parameters(p0[3:], "iso", False)
        assert (len(p) == Nstars*5) == True
        assert (Nstars == N) == True
        p, N = gc.transform_parameters(p0, "iso", True)
        assert (len(p == Nstars*5))
        assert (Nstars == N) == True
        pars, lnages = gc.transform_parameters(p0[:6], "gyro", False)
        assert (len(pars) == 3) == True
        assert (len(lnages) == Nstars) == True
        pars, lnages = gc.transform_parameters(p0, "gyro", True)
        assert (len(pars) == 3) == True
        assert (len(lnages) == Nstars) == True

    def test_iso_lnprior_one_star(self):
        """
        Test the iso_lnprior function works on one star.
        """
        print("Test 6")
        one_star_pars = [np.log(1.), np.log(4.), 0., np.log(1000.), 0.]
        assert (np.isfinite(gc.iso_lnprior(np.array(one_star_pars)))) \
            == True

    def test_mh_correct_size(self):
        """
        Test the Metropolis Hastings function returns the right size of
        samples.
        """
        print("Test 7")
        i, g, star_number = True, True, None
        DATA_DIR = "/Users/ruthangus/projects/chronometer/chronometer/data"
        d = pd.read_csv(os.path.join(DATA_DIR, "data_file.csv"))
        params, mods = gc.pars_and_mods(DATA_DIR)
        t = [.01, .01, .01, .03, .1, .1, .3, .3, .3, .1, .2, .2, .02, .2, .2,
             .01, .2, .2]
        p0, args = gc.assign_args(params, mods, t, d, i, g,
                                  star_number, verbose=False)
        N, nd = 100, 3 + 5*3
        samples, par, probs = gc.MH(p0, N, .01, *args)
        nsteps, ndim = np.shape(samples)
        assert (nsteps == N) == True
        assert (ndim == nd) == True

    def test_run_MCMC_sample_shape(self):
        print("Test 8")
        DATA_DIR = "/Users/ruthangus/projects/chronometer/chronometer/data"
        d = pd.read_csv(os.path.join(DATA_DIR, "data_file.csv"))
        p0, mods = gc.pars_and_mods(DATA_DIR)
        samps, last_samp, probs = gc.run_MCMC(p0, mods, d,
                                              True, True, None, 10, 1e-2)
        assert (np.shape(samps) == (10, 18)) == True
        assert (len(last_samp) == 18) == True
        samps, last_samp, probs = gc.run_MCMC(p0, mods, d,
                                              True, False, 0, 10, 1e-2)
        assert (np.shape(samps) == (10, 5)) == True
        assert (len(last_samp) == 5) == True
        samps, last_samp, probs = gc.run_MCMC(p0, mods, d,
                                              False, True, None, 10, 1e-2)
        assert (np.shape(samps) == (10, 6)) == True
        assert (len(last_samp) == 6) == True

    def test_gibbs_control(self):
        print("Test 9")
        DATA_DIR = "/Users/ruthangus/projects/chronometer/chronometer/data"
        d = pd.read_csv(os.path.join(DATA_DIR, "data_file.csv"))
        p0, mods = gc.pars_and_mods(DATA_DIR)
        nsteps, niter = 5, 1
        samples, lnprobs = gc.gibbs_control(p0, mods, d,
                                            nsteps, niter, 1e-2)
        print(np.shape(samples))
        print(np.shape(lnprobs))

    def test_iso_lnlike_one_star(self):
        """
        Test the iso_lnlike function works on one star.
        """
        DATA_DIR = "/Users/ruthangus/projects/chronometer/chronometer/data"
        d = pd.read_csv(os.path.join(DATA_DIR, "data_file.csv"))
        p0, mods = gc.pars_and_mods(DATA_DIR)
        print("Test 10")
        t = [.01, .01, .01, .03, .1, .1, .3, .3, .3, .1, .2, .2, .02, .2, .2,
             .01, .2, .2]
        params, args = gc.assign_args(p0, mods, t, d, True,
                                      False, 0, verbose=False)
        assert (np.isfinite(gc.iso_lnlike(params, args[0],
                                          all_params=False))) == True


    def test_MH_step_prob_increase(self):
        DATA_DIR = "/Users/ruthangus/projects/chronometer/chronometer/data"
        d = pd.read_csv(os.path.join(DATA_DIR, "data_file.csv"))
        p0, mods = gc.pars_and_mods(DATA_DIR)
        t = [.01, .01, .01, .03, .1, .1, .3, .3, .3, .1, .2, .2, .02, .2, .2,
             .01, .2, .2]
        params, args = gc.assign_args(p0, mods, t, d, True,
                                      True, None, verbose=False)
        params, old_lnprob, accept = gc.MH_step(params, len(params), 0.,
                                                *args)
        params[6] = np.log(5)
        par, new_lnprob, accept = gc.MH_step(params, len(params), 0., *args)
        assert (new_lnprob < old_lnprob) == True
