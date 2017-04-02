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


def gc_model(ln_age, bv):
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
    a, b, n = [.7725, .601, .5189]
    return a*(np.exp(ln_age)*1e3)**n * (bv - .4)**b


def gc_lnlike(params, period, period_errs, bv, bv_errs):
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
    model_periods = gc_model(params, bv)
    return sum(-.5*((period - model_periods)/period_errs)**2)


def lnprior(params):
    N = int(len(params[3:])/5)  # number of stars
    ln_age = params[3+N:3+2*N]  # parameter assignment
    age_prior = sum([np.log(priors.age_prior(np.log10(1e9*np.exp(i))))
                     for i in ln_age])
    m = (-10 < params) * (params < 10)  # Broad bounds on all params.
    if sum(m) == len(m):
        return age_prior
    else:
        return -np.inf


def lnprob(params, period, period_errs, bv, bv_errs):
    """
    The joint log-probability of age given gyro and iso parameters.
    mod: (list)
        list of pre-computed star model objects.
    gyro: (bool)
        If True, the gyro likelihood will be used.
    iso: (bool)
        If True, the iso likelihood will be used.
    """
    return gc_lnlike(params, period, period_errs, bv, bv_errs) + \
        lnprior(params)


if __name__ == "__main__":

    # The parameters
    p0 = np.array([4.56, .5])

    periods = np.array([26., 8.3])
    period_errs = np.array([.1, .1])
    bvs = np.array([.65, .65])
    bv_errs = np.array([.01, .01])

    # Plot the data
    # xs = np.linspace(.1, 6, 100)
    # ps = gc_model(p0[:3], np.log(xs), p0[3], bvs[0])
    # plt.clf()
    # plt.plot(ages, periods, "k.")
    # plt.plot(xs, ps)
    # plt.xlabel("Age (Gyr)")
    # plt.ylabel("Period (days)")
    # plt.savefig("period_age_data")

    # test the gyro lhf
    print("gyro_lnlike = ", gc_lnlike(p0, periods, period_errs, bvs, bv_errs))

    # test the lnprob.
    print("lnprob = ", lnprob(p0, periods, period_errs, bvs, bv_errs))

    start = time.time()

    # Run emcee and plot corner
    nwalkers, nsteps, ndim = 64, 10000, len(p0)
    p0 = [1e-4*np.random.rand(ndim) + p0 for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                    args=[periods, period_errs, bvs, bv_errs])
    print("burning in...")
    pos, _, _ = sampler.run_mcmc(p0, 2000)
    sampler.reset()
    print("production run...")
    sampler.run_mcmc(pos, nsteps)

    end = time.time()
    print("Time taken = ", (end - start)/60., "mins")

    flat = np.reshape(sampler.chain, (nwalkers*nsteps, ndim))
    fig = corner.corner(flat)
    fig.savefig("corner_test")

    # Plot probability
    plt.clf()
    plt.plot(sampler.lnprobability.T, "k")
    plt.savefig("prob_trace")

    # Plot chains
    for i in range(ndim):
        plt.clf()
        plt.plot(sampler.chain[:, :,  i].T, alpha=.5)
        plt.savefig("{}_trace".format(i))
