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


def gc_lnlike(params, period, period_errs, bv, bv_errs, all_params=False):
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
    if all_params:
        N = int((len(params) - 3)/5)
        pars = params[:3]
        ln_ages = params[3:3+N]
    else:
        pars = params[:3]
        ln_ages = params[3:]
    model_periods = gc_model(pars, ln_ages, bv)
    return sum(-.5*((period - model_periods)/period_errs)**2)


def iso_lnlike(lnparams, mods, all_params=False):
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
    if all_params:
        p = lnparams[3:]
    else:
        p = lnparams*1
    N = int(len(p)/5)

    # Transform to linear space
    # mass, age, feh, distance, Av
    p[:N] = np.exp(p[:N])
    p[N:2*N] = np.log10(1e9*np.exp(p[N:2*N]))
    p[3*N:4*N] = np.exp(p[3*N:4*N])

    ll = [mods[i].lnlike(p[i::N]) for i in range(len((mods)))]
    return sum(ll)


def gyro_lnprior(params):
    m = (-10 < params) * (params < 10)  # Broad bounds on all params.
    if sum(m) == len(m):
        return 0.
    else:
        return -np.inf


def iso_lnprior(params):
    N = int(len(params)/5)  # number of stars
    ln_age = params[N:2*N]  # parameter assignment
    ln_mass = params[2*N:3*N]
    feh = params[3*N:4*N]
    d = params[4*N:5*N]
    Av = params[5*N:6*N]
    age_prior = sum([np.log(priors.age_prior(np.log10(1e9*np.exp(i))))
                     for i in ln_age])
    feh_prior = sum([np.log(priors.feh_prior(i)) for i in feh])
    distance_prior = sum([np.log(priors.distance_prior(np.exp(i))) for i
                          in d])
    Av_prior = sum([np.log(priors.AV_prior(Av[i])) for i in Av])
    m = (-10 < params) * (params < 10)  # Broad bounds on all params.
    if sum(m) == len(m):
        return age_prior + feh_prior + distance_prior + Av_prior
    else:
        return -np.inf


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
    print("params = ", params)
    m = (-20 < params) * (params < 20)  # Broad bounds on all params.
    print(m)
    if sum(m) == len(m):
        print("yes")
        return age_prior + feh_prior + distance_prior + Av_prior
    else:
        print("No")
        return -np.inf


def lnprob(params, *args):
    """
    The joint log-probability of age given gyro and iso parameters.
    params: (array)
        The parameter array.
    mod: (list)
        list of pre-computed star model objects.
    """
    if len(params) < 6:
        iso, gyro = False, True  # GYRO
    elif len(params) % 5 == 0:
        iso, gyro = True, False  # ISO
    else:
        iso, gyro = True, True

    if iso and gyro:
        mods, period, period_errs, bv, bv_errs = args
        return gc_lnlike(params, period, period_errs, bv, bv_errs,
                         all_params=True) + \
            iso_lnlike(params, mods, all_params=True) + lnprior(params)
    elif gyro:
        period, period_errs, bv, bv_errs = args
        return gc_lnlike(params, period, period_errs, bv, bv_errs) + \
            gyro_lnprior(params)
    elif iso:
        mods = args[0]
        return iso_lnlike(params, mods) + iso_lnprior(params)


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


    # iso_lnlike preamble.
    mist = MIST_Isochrone()
    mods = []
    for i in range(len(periods)):
        mods.append(StarModel(mist, J=(Js[i], J_errs[i]),
                              H=(Hs[i], H_errs[i]), K=(Ks[i], K_errs[i]),
                              use_emcee=True))

    start = time.time()

    # Run emcee and plot corner
    i, g = True, False
    if g and i:
        print("gyro and iso")
        args = [mods, periods, period_errs, bvs, bv_errs]
        truths = [.7725, .601, .5189, np.log(4.56), np.log(.5), np.log(1),
                  np.log(1), 0., 0., np.log(10), np.log(10), 0., 0.]
        labels = ["$a$", "$b$", "$n$", "$\ln(Age_1)$", "$\ln(Age_2)$",
                  "$\ln(Mass_1)$", "$\ln(Mass_2)$", "$[Fe/H]_1$",
                  "$[Fe/H]_2$", "$\ln(D_1)$", "$\ln(D_2)$", "$A_v1$",
                  "$A_v2$"]
    if g and not i:  # If gyro inference
        print("gyro")
        p0 = gyro_p0*1
        args = [periods, period_errs, bvs, bv_errs]
        truths = [.7725, .601, .5189, np.log(4.56), np.log(.5)]
        labels = ["$a$", "$b$", "$n$", "$\ln(Age_1)$", "$\ln(Age_2)$"]
    elif i and not g:  # If Iso inference.
        print("iso")
        p0 = iso_p0*1
        args = [mods]
        truths = [np.log(4.56), np.log(.5), np.log(1), np.log(1), 0., 0.,
                np.log(10), np.log(10), 0., 0.]
        labels = ["$\ln(Age_1)$", "$\ln(Age_2)$", "$\ln(Mass_1)$",
                "$\ln(Mass_2)$", "$[Fe/H]_1$", "$[Fe/H]_2$", "$\ln(D_1)$",
                "$\ln(D_2)$", "$A_v1$", "$A_v2$"]

    print("p0 = ", p0)
    nwalkers, nsteps, ndim = 64, 10000, len(p0)
    p0 = [1e-4*np.random.rand(ndim) + p0 for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=args)
    print("burning in...")
    pos, _, _ = sampler.run_mcmc(p0, 2000)
    sampler.reset()
    print("production run...")
    sampler.run_mcmc(pos, nsteps)

    end = time.time()
    print("Time taken = ", (end - start)/60., "mins")

    flat = np.reshape(sampler.chain, (nwalkers*nsteps, ndim))
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
        # plt.ylabel(labels[i])
        plt.savefig("{}_trace".format(i))
