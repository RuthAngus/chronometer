# Assembling likelihood functions for Gyrochronology, isochrones, dynamics
# and asteroseismology.

import os

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
    return a*(age*1e3)**n * (bv[0] - .4)**b


def gc_lnlike(params, data, data_err):
    """
    Probability of age and model parameters given rotation period and colour.
    parameters:
    ----------
    params: (array)
        The array of log parameters: a, b, n, age.
    data: (array)
        an array containing rotation periods and B-V colour.
    data_err: (array)
        an array containing rotation period and B-V uncertainties.
    """
    periods, bv = data
    p_err, bv_err = data
    model_periods = gc_model(params, bv)
    return sum(-.5*((periods - model_periods)/p_err)**2)


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


def iso_lnlike(params, bands, parallax, fit_parallax=True):
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

    bands = data
    if fit_parallax:
        args = dict(bands, parallax=(parallax))
    else:
        args = bands
    mod = StarModel(mist, **args)
    print(mod.lnlike(params))
    return mod.lnlike(params)


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

    # test on the Sun first.
    MJ = 3.711  # absolute magnitudes
    MH = 3.453
    MK = 3.357
    J = distance_modulus(MJ, 10)
    H = distance_modulus(MH, 10)
    K = distance_modulus(MK, 10)

    bands = dict(J=(J, .01), H=(H, .01), K=(K, .01),)
    parallax = (.1, .001)
    period = (26., 1.)
    bv = (.65, .01)
    args = (bands, parallax, period, bv)

    # test the gyro model
    gyro_params = np.log(np.array([a, b, n, 4.56]))
    print(gc_model(gyro_params, bv))


    DATA_DIR = "/Users/ruthangus/projects/chronometer/chronometer/data"
    # load data.
    cs = pd.read_csv(os.path.join(DATA_DIR, "clusters.csv"))
    m = (.6 < cs.bv.values) * (cs.bv.values < .7)
    cs = cs.iloc[m]
    periods = cs.period.values
    bvs = cs.bv.values
    ages = cs.age.values * 1e3
    age_err = cs.age_err.values
