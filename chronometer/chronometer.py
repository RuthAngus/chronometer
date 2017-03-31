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
from models import gc_model


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
    N = int(len(params[3:])/5)  # number of stars
    ln_age = params[3+N:3+2*N]  # parameter assignment
    model_periods = gc_model(params[:3], ln_age, bv[0])
    return sum(-.5*((period[0] - model_periods)/period[1])**2)


def iso_lnlike(lnparams, mods):
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
    p = lnparams[3:]*1
    N = int(len(p)/5)

    # Transform to linear space
    p[:N] = np.exp(p[:N])
    p[N:2*N] = np.log10(1e9*np.exp(p[N:2*N]))
    p[3*N:4*N] = np.exp(p[3*N:4*N])

    ll = [mods[i].lnlike(p[i::N]) for i in range(len((mods)))]
    return sum(ll)


def lnprior(params):
    N = int(len(params[3:])/5)  # number of stars
    ln_age = params[3+N:3+2*N]  # parameter assignment
    ln_mass = params[3+2*N:3+3*N]
    feh = params[3+3*N:3+4*N]
    d = params[3+4*N:3+5*N]
    Av = params[3+5*N:3+6*N]
    age_prior = sum([np.log(priors.age_prior(np.log10(1e9*np.exp(i))))
                     for i in ln_age])
    feh_prior = sum([np.log(priors.feh_prior(i)) for i in feh])
    distance_prior = sum([np.log(priors.distance_prior(np.exp(i))) for i
                          in d])
    Av_prior = sum([np.log(priors.AV_prior(Av[i])) for i in Av])
    m = (-10 < params) * (params < 10)
    if sum(m) == len(m):
        return age_prior + feh_prior + distance_prior + Av_prior
    else:
        return -np.inf


def lnprob(params, mods, period, bv, gyro=True, iso=False):
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
        return iso_lnlike(params, mods) + gc_lnlike(params, period, bv) + \
            lnprior(params)
    elif gyro:
        return gc_lnlike(params, period, bv) + lnprior(params)
    elif iso:
        return iso_lnlike(params, mods) + lnprior(params)


if __name__ == "__main__":

    # The parameters
    gc = np.array([.7725, .601, .5189])
    ages = np.array([4.56, .5])
    masses = np.array([1., 1.])
    fehs = np.array([0., 0.])
    ds = np.array([10., 10.])
    Avs = np.array([0., 0.])
    p0 = np.concatenate((gc, np.log(masses), np.log(ages), fehs, np.log(ds),
                         Avs))

    # test on the Sun at 10 pc first.
    J, J_err = 3.711, .01  # absolute magnitudes/apparent at D = 10pc
    H, H_err = 3.453, .01
    K, K_err = 3.357, .01

    # The data
    Js = np.array([J, J])
    J_errs = np.array([J_err, J_err])
    Hs = np.array([H, H])
    H_errs = np.array([H_err, H_err])
    Ks = np.array([K, K])
    K_errs = np.array([K_err, K_err])
    parallaxes = np.array([.1*1e3, .1*1e3])
    parallax_errs = np.array([.001, .001])
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
    print("gyro_lnlike = ", gc_lnlike(p0, periods, bvs))

    # test the iso_lnlike
    mist = MIST_Isochrone()

    # iso_lnlike preamble.
    start = time.time()
    mods = []
    for i in range(len(periods)):
        mods.append(StarModel(mist, J=(Js[i], J_errs[i]),
                              H=(Hs[i], H_errs[i]), K=(Ks[i], K_errs[i]),
                              use_emcee=True))

        start = time.time()
        mods[i].lnlike(p0)
        end = time.time()
        print("preamble time = ", end - start)

        start = time.time()
        print("iso_lnlike = ", iso_lnlike(p0, mods))
        end = time.time()
        print("lhf time = ", end - start)

        # test the lnprob.
        print("lnprob = ", lnprob(p0, mods, periods, bvs, gyro=True,
                                  iso=True))
    start = time.time()

    # Run emcee and plot corner
    nwalkers, nsteps, ndim = 64, 5000, len(p0)
    p0 = [1e-4*np.random.rand(ndim) + p0 for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                    args=[mods, periods, bvs])
    print("burning in...")
    pos, _, _ = sampler.run_mcmc(p0, 1000)
    sampler.reset()
    print("production run...")
    sampler.run_mcmc(pos, nsteps)

    end = time.time()
    print("Time taken = ", (end - start)/60., "mins")

    flat = np.reshape(sampler.chain, (nwalkers*nsteps, ndim))
    truths = [.7725, .601, .5189, np.log(4.56), np.log(.5), np.log(1),
              np.log(1), 0., 0., np.log(10), np.log(10), 0., 0.]
    labels = ["$a$", "$b$", "$n$", "$\ln(Age_1)$", "$\ln(Age_2)$",
              "$\ln(Mass_1)$", "$\ln(Mass_2)$", "$[Fe/H]_1$", "$[Fe/H]_2$",
              "$\ln(D_1)$", "$\ln(D_2)$", "$A_v1$", "$A_v2$"]
    fig = corner.corner(flat, labels=labels, truths=truths)
    fig.savefig("corner_test")

    # Plot probability
    plt.clf()
    plt.plot(sampler.lnprobability.T, "k")
    plt.savefig("prob_trace")

    # Plot chains
    for i in range(ndim):
        plt.clf()
        plt.plot(sampler.chain[:, :,  i].T, alpha=.5)
        plt.ylabel(labels[i])
        plt.savefig("{}_trace".format(i))
