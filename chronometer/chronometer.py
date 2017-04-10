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

import teff_bv as tbv
from utils import replace_nans_with_inits, vk2teff, make_param_dict

plotpar = {'axes.labelsize': 18,
           'font.size': 10,
           'legend.fontsize': 18,
           'xtick.labelsize': 18,
           'ytick.labelsize': 18,
           'text.usetex': True}
plt.rcParams.update(plotpar)


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
        p = lnparams[3:]*1
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


def lnprior(params):
    """
    lnprior on the parameters when both iso and gyro parameters are being
    inferred.
    """
    # Parameter assignment
    N = int(len(params[3:])/5)  # number of stars
    ln_mass = params[3:3+N]  # parameter assignment
    ln_age = params[3+N:3+2*N]
    feh = params[3+2*N:3+3*N]
    distance = params[3+3*N:3+4*N]
    Av = params[3+4*N:3+5*N]

    # Calculate priors from Tim's isochrones.py
    age_prior = sum([np.log(priors.age_prior(np.log10(1e9*np.exp(i))))
                     for i in ln_age])
    feh_prior = sum([np.log(priors.feh_prior(i)) for i in feh])
    distance_prior = sum([np.log(priors.distance_prior(np.exp(i))) for i
                          in distance])
    # Av_prior = sum([np.log(priors.AV_prior(Av[i])) for i in Av])
    g_prior = priors.lng_prior(params[:3])
    m = (-20 < params) * (params < 20)  # Broad bounds on all params.
    mAv = (0 < Av) * (Av < 1)
    if sum(m) == len(m) and sum(mAv) == len(mAv):
        return age_prior + feh_prior + distance_prior + g_prior #+ Av_prior
    else:
        return -np.inf


def lnprob(params, *args):
    """
    The joint log-probability of age given gyro and iso parameters.
    params: (array)
        The parameter array.
    mod: (list)
        list of pre-computed star model objects.
    """

    mods, period, period_errs, bv, bv_errs = args
    return gc_lnlike(params, period, period_errs, bv, bv_errs,
                        all_params=True) + \
        iso_lnlike(params, mods, all_params=True) + lnprior(params)


def plot_gyro_result(d, mcmc_result):
    # Plot the gyro result
    xs = np.linspace(.1, 6, 100)
    ps0 = gc_model(mcmc_result[:3], np.log(xs), .65)
    ps1 = gc_model(mcmc_result[:3], np.log(xs), .65)
    N = len(d.age.values)
    age_results = np.exp(mcmc_result[3+N:3+2*N])

    plt.clf()
    plt.plot(d.age.values, d.period.values, "k.")
    plt.plot(age_results, d.period.values, "m.")
    plt.plot(xs, ps0, label="$\mathrm{Before}$")
    plt.plot(xs, ps1, label="$\mathrm{After}$")
    plt.legend()
    plt.xlabel("$\mathrm{Age~(Gyr)}$")
    plt.ylabel("$\mathrm{Period~(days)}$")
    plt.savefig("period_age_data")


def get_inits(d):
    # The parameters
    gc = np.array([.7725, .601, .5189])
    d = replace_nans_with_inits(d)
    p0 = np.concatenate((gc, np.log(d.mass.values),
                         np.log(d.age.values), d.feh.values,
                         np.log(1./d.parallax.values*1e3),
                         d.Av.values))
    return p0


def get_mod_list(d):
    # iso_lnlike preamble - make a list of 'mod' objects: one for each star.
    mist = MIST_Isochrone()
    mods = []
    for i in range(len(d.period.values)):
        param_dict = make_param_dict(d, i)

        # Remove missing parameters from the dict.
        param_dict = {k: param_dict[k] for k in param_dict if
                      np.isfinite(param_dict[k]).all()}
        mods.append(StarModel(mist, **param_dict))
    return mods


if __name__ == "__main__":

    DATA_DIR = "/Users/ruthangus/projects/chronometer/chronometer/data"
    d = pd.read_csv(os.path.join(DATA_DIR, "data_file.csv"))

    p0 = get_inits(d)
    mods = get_mod_list(d)
    args = [mods, d.period.values, d.period_err.values, d.bv.values,
            d.bv_err.values]

    start = time.time()  # timeit

    # Run emcee and plot corner
    print("p0 = ", p0)
    nwalkers, nsteps, ndim = 64, 5000, len(p0)
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
    flat = flat[-1000:, :]
    fig = corner.corner(flat)
    fig.savefig("corner")
    mcmc_result = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                    zip(*np.percentile(flat, [16, 50, 84], axis=0)))
    mcmc_result = np.percentile(flat, 50, axis=0)
    print("results = ", mcmc_result)
    N = len(d.age.values)
    print("Ages = ", np.exp(mcmc_result[3+N:3+2*N]))

    print("Plotting results and traces...")
    plot_gyro_result(d, mcmc_result)
