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
    # return (np.log(p) - np.log(a) - b*np.log(bv - .4))/n
    return n*np.log(age) + np.log(a) + b*np.log(bv - .4)


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


def fit_star():
    bands = dict(
            J=(star.jmag, star.jmag_err),
            H=(star.hmag, star.hmag_err),
            K=(star.kmag, star.kmag_err),
        )
    args = (bands, parallax, period, bv)


if __name__ == "__main__":

    # Testing the isochrones likelihood function.
    mist = MIST_Isochrone()
    mod = StarModel(mist, Teff=(5700, 100), logg=(4.5, 0.1), feh=(0.0, 0.1))

    import time
    start = time.time()
    # mass, age, metalicity, distance, extinction
    p0 = [0.8, 9.5, 0.0, 200, 0.2]
    print(mod.lnlike(p0))
    end = time.time()
    print(end - start, "seconds")

    start = time.time()
    p0 = [0.7, 9.5, 0.0, 200, 0.2]
    print(mod.lnlike(p0))
    end = time.time()
    print(end - start, "seconds")

    start = time.time()
    p0 = [0.6, 9.5, 0.0, 200, 0.2]
    print(mod.lnlike(p0))
    end = time.time()
    print(end - start, "seconds")
    assert 0

    DATA_DIR = "/Users/ruthangus/projects/chronometer/chronometer/data"

    params = np.log([.7725, .601, .5189])

    # load data.
    cs = pd.read_csv(os.path.join(DATA_DIR, "clusters.csv"))
    m = (.6 < cs.bv.values) * (cs.bv.values < .7)
    cs = cs.iloc[m]
    periods = cs.period.values
    bvs = cs.bv.values
    ages = cs.age.values * 1e3
    age_err = cs.age_err.values

    print(gc_lnlike(params, [periods, bvs], ages, age_err))

    # Run emcee
    ndim, nwalkers, nsteps = len(params), 24, 10000
    p0 = [1e-4*np.random.rand(ndim) + params for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, gc_lnprob,
                                    args=[[periods, bvs], ages, age_err])
    pos, _, _ = sampler.run_mcmc(p0, 5000)
    sampler.reset()
    sampler.run_mcmc(pos, nsteps)
    flat = np.reshape(sampler.chain, (nwalkers*nsteps, ndim))
    fig = corner.corner(np.exp(flat), labels=["a", "b", "n"])
    fig.savefig("corner_test")
    results = [np.median(flat[:, 0]), np.median(flat[:, 1]),
               np.median(flat[:, 2])]
    print(results)

    # Plot data with model
    xs = np.linspace(min(periods), max(periods), 100)
    init = gc_model(params, [xs, .65])
    fit = gc_model(results, [xs, .65])
    plt.clf()
    plt.plot(init, np.log(xs), label="init")
    plt.plot(fit, np.log(xs), label="fit")
    plt.legend()
    plt.plot(np.log(ages), np.log(periods), "k.")
    plt.xlabel("age")
    plt.ylabel("period")
    plt.savefig("model")
