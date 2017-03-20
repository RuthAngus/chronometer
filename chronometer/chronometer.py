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
import priors


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
    model_periods = gc_model(params[:4], bv[0])
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
    a, b, n, lnage, lnmass, feh, lnd, Av = lnparams
    params = np.array([np.exp(lnmass), np.log10(1e9*np.exp(lnage)), feh,
                       np.exp(lnd), Av])
    return mod.lnlike(params)


def lnprior(params):
    age_prior = np.log(priors.age_prior(np.log10(1e9*np.exp(params[3]))))
    feh_prior = np.log(priors.feh_prior(params[5]))
    distance_prior = np.log(priors.distance_prior(np.exp(params[6])))
    Av_prior = np.log(priors.AV_prior(params[7]))
    if -10 < params[0] < 10 and -10 < params[1] < 10 and \
            -10 < params[2] < 10:
        return age_prior + feh_prior + distance_prior + Av_prior
    else:
        return -np.inf


def lnprob(params, mod, period, bv, gyro=True, iso=True):
    """
    The joint log-probability of age given gyro and iso parameters.
    mod: (list)
        list of pre-computed star model objects.
    gyro: (bool)
        If True, the gyro likelihood will be used.
    iso: (bool)
        If True, the iso likelihood will be used.
    """

    if gyro and iso:
        return iso_lnlike(params, mod) + gc_lnlike(params, period, bv) + \
            lnprior(params)
    elif gyro:
        return gc_lnlike(params, period, bv) + lnprior(params)
    elif iso:
        return iso_lnlike(params, mod) + lnprior(params)


def probtest(xs, i):
    lps = []
    p = params + 0
    for x in xs:
        p[i] = x
        lp = lnprob(p, mod, period, bv, gyro=False, iso=True)
        lps.append(lp)
    plt.clf()
    plt.plot(xs, lps)
    plt.xlabel("X")
    plt.ylabel("lnprob")
    plt.savefig("probs_{}".format(i))


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
    print("gyro_lnlike = ", gc_lnlike(gyro_params, period, bv))

    # test the iso_lnlike
    mist = MIST_Isochrone()
    iso_params = np.array([np.log(mass), np.log(age), feh, np.log(d), Av])

    # iso_lnlike preamble.
    start = time.time()
    mod = StarModel(mist, J=(J, J_err), H=(H, H_err), K=(K, K_err),
                    parallax=(.1, .001))
    p0 = np.array([params[0], params[1], params[2], params[3], params[4]])

    start = time.time()
    mod.lnlike(p0)
    end = time.time()
    print("preamble time = ", end - start)

    start = time.time()
    print("iso_lnlike = ", iso_lnlike(params, mod))
    end = time.time()
    print("lhf time = ", end - start)

    # test the lnprob.
    print("lnprob = ", lnprob(params, mod, period, bv, gyro=True, iso=True))

    ages = np.log(np.arange(1., 10., 1))
    masses = np.log(np.arange(.1, 2., .1))
    fehs = np.arange(-.1, .1, .01)
    ds = np.log(np.arange(8, 20, 1))
    Avs = np.arange(.1, .5, .01)
    probtest(ages, 3)
    probtest(masses, 4)
    probtest(fehs, 5)
    probtest(ds, 6)
    probtest(Avs, 7)

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
    truths = [.7725, .601, .5189, np.log(4.56), np.log(1), 0., np.log(10),
              0.]
    labels = ["$a$", "$b$", "$n$", "$\ln(Age)$", "$\ln(Mass)$", "$[Fe/H]$",
              "$\ln(D)$", "$A_v$"]
    fig = corner.corner(flat, labels=labels, truths=truths)
    fig.savefig("corner_test")

    plt.clf()
    print(np.shape(sampler.lnprobability.T))
    plt.plot(sampler.lnprobability.T, "k")
    plt.savefig("prob_trace")

    for i in range(ndim):
        plt.clf()
        plt.plot(sampler.chain[:, :,  i].T, alpha=.5)
        plt.ylabel(labels[i])
        plt.savefig("{}_trace".format(i))
