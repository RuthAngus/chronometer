"""
Test MH sampling on one star.
"""

import os
import sys

import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from isochrones import StarModel
from isochrones.mist import MIST_Isochrone
import h5py

import corner
import priors
from models import gc_model, gyro_model
import emcee

from utils import replace_nans_with_inits, vk2teff, make_param_dict, \
    parameter_assignment, pars_and_mods, transform_parameters

plotpar = {'axes.labelsize': 18,
           'font.size': 10,
           'legend.fontsize': 18,
           'xtick.labelsize': 18,
           'ytick.labelsize': 18,
           'text.usetex': True}
plt.rcParams.update(plotpar)


def lnlike(params, mod):
    """
    Probability of age and model parameters given rotation period and colour.
    parameters:
    ----------
    params: (array)
        The array of log parameters: a, b, n, age.
    par_inds: (array)
        The parameters to vary.
    """
    print(params, mod)
    p = params*1
    p[0] = np.exp(p[0])
    p[1] = np.log10(1e9*np.exp(p[1]))
    p[3] = np.exp(p[3])
    return mod.lnlike(p)


def lnprior(params):
    """
    lnprior on all parameters.
    """
    age_prior = np.log(priors.age_prior(np.log10(1e9*np.exp(params[1]))))
    feh_prior = np.log(priors.feh_prior(params[2]))
    distance_prior = np.log(priors.distance_prior(np.exp(params[3])))

    m = (-20 < params) * (params < 20)  # Broad bounds on all params.
    mAv = (0 <= params[4]) * (params[4] < 1)

    if sum(m) == len(m) and mAv:
        return age_prior + feh_prior + distance_prior
    else:
        return -np.inf


def lnprob(params, mod):
    """
    The joint log-probability of age given gyro and iso parameters.
    params: (array)
        The parameter array.
    mod: (list)
        list of pre-computed star model objects.
    """
    print(lnlike(params, mod), lnprior(params))
    input("e")
    return lnlike(params, mod) + lnprior(params)


def MH(par, lnprob, nsteps, t, mod, emc=False):
    """
    This is where the full list of parameters is reduced to just those being
    sampled.
    params:
    -------
    par: (list)
        The parameters.
    nsteps: (int)
        Number of samples.
    t: (float)
        The std of the proposal distribution.
    args:
    x, y, yerr: (arrays)
        The data
    """
    samples = np.zeros((nsteps, len(par)))

    if emc:
        nwalkers, ndim = 64, len(par)
        p0 = [1e-4*np.random.rand(ndim) + par for i in range(nwalkers)]
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[mod])
        par, _, _ = sampler.run_mcmc(p0, nsteps)
        samples = np.reshape(sampler.chain, (nwalkers * nsteps, ndim))
        probs = sampler.lnprobability.T[:, 0]
        accept = 0

    else:
        accept, probs = 0, []
        for i in range(nsteps):
            par, new_prob, acc = MH_step(par, lnprob, t, mod)
            accept += acc
            probs.append(new_prob)
            samples[i, :] = par

    print("Acceptance fraction = ", accept/float(nsteps))
    return samples, par, probs


def MH_step(par, lnprob, t, *args, emc=False):
    if emc:
        nwalkers, ndim = 10, len(par)
        p0 = [par + np.random.multivariate_normal(np.zeros((len(par))), t)
              for i in range(nwalkers)]
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=args)
        sampler.run_mcmc(p0, 10)
        return sampler.chain[0][-1], sampler.lnprobability.T[-1, 0], 0
    newp = par + np.random.multivariate_normal(np.zeros((len(par))), t)
    new_lnprob = lnprob(newp, *args)
    alpha = np.exp(new_lnprob - lnprob(par, *args))
    if alpha > 1:
        par = newp*1
        accept = 1
    else:
        u = np.random.uniform(0, 1)
        if alpha > u:
            par = newp*1
            accept = 1
        else:
            accept = 0
            new_lnprob = lnprob(par, *args)
    return par, new_lnprob, accept


if __name__ == "__main__":

    DATA_DIR = "/Users/ruthangus/projects/chronometer/chronometer/data"
    d = pd.read_csv(os.path.join(DATA_DIR, "data_file.csv"))

    # Generate the initial parameter array and the mods objects from the data.
    params, mods = pars_and_mods(DATA_DIR)
    params = params[3:]
    params = params[::3]
    par_inds = np.array([3, 6, 9, 12, 15])
    mods = mods[0]

    lnprob(params, mods)
    input('e')

    nsteps = 10000

    with h5py.File("emcee_posterior_samples.h5", "r") as f:
        samples = f["samples"][...]
    ts = np.cov(samples, rowvar=False)
    t = np.zeros((len(par_inds), len(par_inds)))
    for j, ind in enumerate(par_inds):
        t[j] = ts[ind][par_inds]

    flat, par, lnprobs = MH(params, lnprob, nsteps, t, mods)

    print("Plotting results and traces")
    plt.clf()
    plt.plot(lnprobs[100:])
    plt.xlabel("Time")
    plt.ylabel("ln (probability)")
    plt.savefig("prob_trace_MH_single")

    print("Making corner plot")
    truths = [np.log(1), np.log(4.56), 0., np.log(10), 0.]
    labels = ["$\ln(Mass)$", "$\ln(Age)$", "$[Fe/H]$", "$\ln(D)$", "$A_{v1}$"]
    fig = corner.corner(flat, truths=truths, labels=labels)
    fig.savefig("corner_MH_single")

    # Plot chains
    ndim = len(params)
    for i in range(ndim):
        plt.clf()
        plt.plot(flat[:, i].T, alpha=.5)
        plt.savefig("{}_trace_MH_single".format(i))
