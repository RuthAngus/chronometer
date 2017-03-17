# Assembling likelihood functions for Gyrochronology, isochrones, dynamics
# and asteroseismology.

import os

import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.misc as spm
from isochrones import StarModel
from isochrones.mist import MIST_Isochrone

from simple_gibbs import gibbs
import emcee
import corner


def gc_model(params, bv):
    """
    Given a B-V colour and an age predict a rotation period.
    Returns log(age) in Myr.
    parameters:
    ----------
    params: (array)
        The array of age and (log) gyro parameters, a, b and n.
    data: (array)
        A an array containing colour.
    """
    a, b, n, age = np.exp(params)
    return a*(age*1e3)**n * (bv - .4)**b


def gc_lnlike(params, period, bv):
    """
    Probability of age and model parameters given rotation period and colour.
    parameters:
    ----------
    params: (array)
        The array of log parameters: a, b, n, age.
    period: (tuple)
        The rotation period and period uncertainty in days
    bv: (tuple)
        The B-V colour and colour uncertainty.
    """
    model_periods = gc_model(params, bv[0])
    return sum(-.5*((period[0] - model_periods)/period[1])**2)


def iso_lnlike(lnparams):
    """
    Some isochronal likelihood function.
    parameters:
    ----------
    params: (array)
        The array of parameters: mass, age, metallicity, distance, extinction.
    bands: (dict)
        The dictionary of Kepler j, h, k with their uncertainties.
    parallax: (tuple)
        The parallax and its uncertainty.
    """
    params = np.array([np.exp(lnparams[0]), np.log10(1e9*np.exp(lnparams[1])),
                      lnparams[2], np.exp(lnparams[3]), lnparams[4]])
    return mod.lnlike(params)


def lnprior(params):
    """
    Some simple, uninformative prior over the parameters.
    a, b, n, age, mass, feh, distance, Av = params
    """
    if -10 < params[0] < 10 and -10 < params[1] < 10 and \
            -10 < params[2] < 10 and -10 < params[3] < 10 and \
            -10 < params[4] < 10 and -10 < params[5] < 10 and \
            -10 < params[6] < 10 and -10 < params[7] < 10:
        return 0.
    else:
        return -np.inf


def lnprob(params, *args, fit_parallax=True, gyro=True,
           iso=True):
    """
    The joint log-probability of age given gyro and iso parameters.
    bands: (dict)
        dictionary of Kepler bands and their uncertainties.
    parallax: (tuple)
        parallax and its uncertainty, etc.
    """
    if fit_parallax:
        bands, parallax, period, bv = args
        isoargs = (bands, parallax)
    else:
        bands, period, bv = args
        isoargs = bands
    gyroargs = (period, bv)

    a, b, n, age, mass, feh, d, Av = params
    iso_params = np.array([mass, age, feh, d, Av])
    gyro_params = np.array([a, b, n, age])

    return iso_lnlike(iso_params, *isoargs) + \
        gc_lnlike(gyro_params, gyro_args) + \
        gc_lnprior(gyro_params) + iso_lnprior(iso_params)


def distance_modulus(M, D):
    return 5*np.log10(D) - 5 + M


if __name__ == "__main__":

    a, b, n = .7725, .601, .5189
    age = 4.56
    mass, feh, d, Av = 1., 0., 10., 0.
    params = np.log(np.array([a, b, n, age, mass, feh, d, Av]))

    _1s = np.ones(10)  # make fake data arrays, length 10.

    # test on the Sun at 10 pc first.
    J, J_err = _1s*3.711, _1s*.01  # absolute magnitudes/apparent at D = 10pc
    H, H_err = _1s*3.453, _1s*.01
    K, K_err = _1s*3.357, _1s*.01

    bands = dict(J=(J, J_err), H=(H, H_err), K=(K, K_err),)
    parallax = (_1s*.1, _1s*.001)
    period = (_1s*26., _1s*1.)
    bv = (_1s*.65, _1s*.01)
    args = (bands, parallax, period, bv)

    # test the gyro model
    gyro_params = np.log(np.array([a, b, n, 4.56]))
    # print(gc_model(gyro_params, bv[0]))

    # test the gyro lhf
    # print(gc_lnlike(gyro_params, period, bv))

    # test the iso_lnlike
    mist = MIST_Isochrone()
    iso_params = np.array([np.log(mass), np.log(age), feh, np.log(d), Av])

    mod = StarModel(mist, J=(J, J_err), H=(H, H_err), K=(K, K_err),
                    parallax=(.1, .001))
    print(iso_lnlike(iso_params))


    DATA_DIR = "/Users/ruthangus/projects/chronometer/chronometer/data"
    # load data.
    cs = pd.read_csv(os.path.join(DATA_DIR, "clusters.csv"))
    m = (.6 < cs.bv.values) * (cs.bv.values < .7)
    cs = cs.iloc[m]
    periods = cs.period.values
    bvs = cs.bv.values
    ages = cs.age.values * 1e3
    age_err = cs.age_err.values
