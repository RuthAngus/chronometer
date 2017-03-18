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
    a, b, n, ln_age = params
    return a*(np.exp(ln_age)*1e3)**n * (bv - .4)**b


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
    return -.5*((period[0] - model_periods)/period[1])**2


def iso_lnlike(lnparams, mod):
    """
    Some isochronal likelihood function.
    parameters:
    ----------
    params: (array)
        The array of parameters: log(mass), log(age in Gyr), metallicity,
        log(distance), extinction.
    mod: (object)
        An isochrones.py starmodel object.
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


def lnprob(params, mod, period, bv, gyro=True, iso=False):
    """
    The joint log-probability of age given gyro and iso parameters.
    mod: (list)
        list of pre-computed star model objects.
    gyro: (bool)
        If True, the gyro likelihood will be used.
    iso: (bool)
        If True, the iso likelihood will be used.
    """
    a, b, n, age, mass, feh, d, Av = params
    iso_params = np.array([mass, age, feh, d, Av])
    gyro_params = np.array([a, b, n, age])

    if gyro and iso:
        return iso_lnlike(iso_params, mod) + gc_lnlike(gyro_params, period,
                                                       bv) + lnprior(params)
    elif gyro:
        return gc_lnlike(gyro_params, period, bv) + lnprior(params)
    elif iso:
        return iso_lnlike(iso_params, mod) + lnprior(params)


def distance_modulus(M, D):
    return 5*np.log10(D) - 5 + M


if __name__ == "__main__":

    a, b, n = .7725, .601, .5189
    age = 4.56
    mass, feh, d, Av = 1., 0., 10., 0.
    params = np.array([a, b, n, np.log(age), np.log(mass), feh, np.log(d),
                       Av])


    # test on the Sun at 10 pc first.
    J, J_err = 3.711, .01  # absolute magnitudes/apparent at D = 10pc
    H, H_err = 3.453, .01
    K, K_err = 3.357, .01

    bands = dict(J=(J, J_err), H=(H, H_err), K=(K, K_err),)
    parallax = (.1, .001)
    period = (26., 1.)
    bv = (.65, .01)

    # test the gyro model
    gyro_params = np.array([a, b, n, np.log(4.56)])
    print(gc_model(gyro_params, bv[0]))

    # test the gyro lhf
    print(gyro_params, period, bv, "1")
    print("gyro_lnlike = ", gc_lnlike(gyro_params, period, bv))

    # test the iso_lnlike
    mist = MIST_Isochrone()
    iso_params = np.array([np.log(mass), np.log(age), feh, np.log(d), Av])

    # iso_lnlike preamble.
    start = time.time()
    mod = StarModel(mist, J=(J, J_err), H=(H, H_err), K=(K, K_err),
                    parallax=(.1, .001))
    p0 = np.array([np.exp(params[0]), np.log10(1e9*np.exp(params[1])),
                      params[2], np.exp(params[3]), params[4]])

    start = time.time()
    mod.lnlike(p0)
    end = time.time()
    print("preamble time = ", end - start)

    start = time.time()
    print("iso_lnlike = ", iso_lnlike(iso_params, mod))
    end = time.time()
    print("lhf time = ", end - start)

    # test the lnprob.
    print("lnprob = ", lnprob(params, mod, period, bv, gyro=True, iso=False))

    nwalkers, nsteps, ndim = 64, 10000, len(params)
    p0 = [1e-4*np.random.rand(ndim) + params for i in range(nwalkers)]

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                    args=[mod, period, bv])
    print("burning in...")
    pos, _, _ = sampler.run_mcmc(p0, 1000)
    sampler.reset()
    print("production run...")
    sampler.run_mcmc(pos, nsteps)
    flat = np.reshape(sampler.chain, (nwalkers*nsteps, ndim))
    fig = corner.corner(flat)
    fig.savefig("corner_test")

    DATA_DIR = "/Users/ruthangus/projects/chronometer/chronometer/data"
    # load data.
    cs = pd.read_csv(os.path.join(DATA_DIR, "clusters.csv"))
    m = (.6 < cs.bv.values) * (cs.bv.values < .7)
    cs = cs.iloc[m]
    periods = cs.period.values
    bvs = cs.bv.values
    ages = cs.age.values * 1e3
    age_err = cs.age_err.values
