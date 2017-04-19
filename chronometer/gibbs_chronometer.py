"""
Now use Gibbs sampling to update individual star parameters and global gyro
parameters.
"""

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
from utils import replace_nans_with_inits, vk2teff, make_param_dict, \
    parameter_assignment, pars_and_mods, transform_parameters

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
    pars, ln_ages = transform_parameters(params, "gyro", all_params)
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
    p, N = transform_parameters(lnparams, "iso", all_params)
    if len(mods) > 1:
        ll = [mods[i].lnlike(p[i::N]) for i in range(len((mods)))]
        return sum(ll)
    else:
        return mods[0].lnlike(p)


def gyro_lnprior(params):
    m = (-10 < params) * (params < 10)  # Broad bounds on all params.
    if sum(m) == len(m):
        return priors.lng_prior(params[:3])
    else:
        return -np.inf


def iso_lnprior(params):
    N, ln_mass, ln_age, feh, ln_distance, Av = parameter_assignment(params,
                                                                    "iso")
    age_prior = sum([np.log(priors.age_prior(np.log10(1e9*np.exp(i))))
                     for i in ln_age])
    feh_prior = sum([np.log(priors.feh_prior(i)) for i in feh])
    distance_prior = sum([np.log(priors.distance_prior(np.exp(i))) for i
                          in ln_distance])
    m = (-20 < params) * (params < 20)
    mAv = (0 <= Av) * (Av < 1)
    if sum(m) == len(m) and sum(mAv) == len(mAv):
        return age_prior + feh_prior + distance_prior # + Av_prior
    else:
        return -np.inf


def lnprior(params):
    """
    lnprior on the parameters when both iso and gyro parameters are being
    inferred.
    """
    N, ln_mass, ln_age, feh, ln_distance, Av = parameter_assignment(params,
                                                                    "both")
    age_prior = sum([np.log(priors.age_prior(np.log10(1e9*np.exp(i))))
                     for i in ln_age])
    feh_prior = sum([np.log(priors.feh_prior(i)) for i in feh])
    distance_prior = sum([np.log(priors.distance_prior(np.exp(i))) for i
                          in ln_distance])
    g_prior = priors.lng_prior(params[:3])
    m = (-20 < params) * (params < 20)  # Broad bounds on all params.
    mAv = (0 <= Av) * (Av < 1)
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

    # Figure out whether the iso or gyro or both likelihoods should be called.
    if args[-1] == "gyro":
        period, period_errs, bv, bv_errs, _ = args
        return gc_lnlike(params, period, period_errs, bv, bv_errs) + \
            gyro_lnprior(params)
    elif args[-1] == "iso":
        mods, _ = args
        return iso_lnlike(params, mods) + iso_lnprior(params)
    elif args[-1] == "both":
        mods, period, period_errs, bv, bv_errs, _ = args
        return gc_lnlike(params, period, period_errs, bv, bv_errs,
                         all_params=True) + \
            iso_lnlike(params, mods, all_params=True) + lnprior(params)


def plot_gyro_result(flat, params_init, i, g):
    # Plot the gyro result
    xs = np.linspace(.1, 6, 100)
    result = [np.median(flat[:, 0]), np.median(flat[:, 1]),
              np.median(flat[:, 2])]

    age_results = np.exp(np.array([np.median(flat[:, 6]),
                                   np.median(flat[:, 7]),
                                   np.median(flat[:, 8])]))

    ps0 = gc_model(params_init[:3], np.log(xs), .65)
    ps1 = gc_model(result, np.log(xs), .65)
    plt.clf()
    plt.plot(d.age.values, d.period.values, "k.", ms=20)
    plt.plot(age_results, d.period.values, "m.", ms=20)
    plt.plot(xs, ps0, label="$\mathrm{Before}$")
    plt.plot(xs, ps1, label="$\mathrm{After}$")
    plt.legend()
    plt.xlabel("$\mathrm{Age~(Gyr)}$")
    plt.ylabel("$\mathrm{Period~(days)}$")
    plt.savefig("period_age_data")


def assign_args(p0, mods, d, i, g, star_number, all_pars=False, verbose=True):
    """
    Create the parameter and mod arrays needed to specify whether global or
    individual parameters will be sampled.
    """
    if g:  # Convert V-K to B-V
        m = np.isfinite(d.bv.values)
        teff = vk2teff(d.vk.values[~m])
        new_bv = tbv.teff2bv(teff, 4.44, 0)
        d.bv.values[~m] = new_bv
        d.bv_err.values[~m] = np.ones(len(d.bv.values[~m])) * .01
    if g and i:
        if verbose:
            print("gyro and iso")
        args = [mods, d.period.values, d.period_err.values, d.bv.values,
                d.bv_err.values, "both"]
    if g and not i:  # If gyro inference
        if verbose:
            print("gyro")
        N = len(d.age.values)
        p0 = np.concatenate((p0[:3], p0[3+N:3+2*N]))
        args = [d.period.values, d.period_err.values, d.bv.values,
                d.bv_err.values, "gyro"]
    elif i and not g:  # If Iso inference.
        if verbose:
            print("iso", star_number)
        if not all_pars:
            p0 = p0[3:]
        if star_number is not None:
            Nstars = int(len(p0)/5.)
            p0 = p0[star_number::Nstars]
            mods = [mods[star_number]]
        args = [mods, "iso"]
    return p0, args


def emc(p0, args, nwalkers, nsteps, burnin):
    """
    Run emcee for a set of parameters.
    Try by first running on individual stars, cycle through.
    Then just run the global parameters, then run all.
    Use emcee for now.
    """
    ndim = len(p0)
    p0 = [1e-4*np.random.rand(ndim) + p0 for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=args)
    print("burning in...")
    pos, _, _ = sampler.run_mcmc(p0, burnin)
    sampler.reset()
    print("production run...")
    sampler.run_mcmc(pos, nsteps)
    end = time.time()
    print("Time taken = ", (end - start)/60., "mins")
    flat = np.reshape(sampler.chain, (nwalkers*nsteps, ndim))
    return flat


def MH(par, nsteps, t, *args):
    """
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
    ndim = len(par)
    samples = np.zeros((nsteps, ndim))
    accept = []
    for i in range(nsteps):
        newp = par + np.random.randn(ndim)*t
        alpha = np.exp(lnprob(newp, *args))/np.exp(lnprob(par, *args))
        if alpha > 1:
            par = newp*1
            accept.append(1)
        else:
            u = np.random.uniform(0, 1)
            if alpha > u:
                par = newp*1
                accept.append(1)
            else:
                accept.append(0)
        samples[i, :] = par
    print("Acceptance fraction = ", float(sum(accept))/len(accept))
    return samples, par


def run_MCMC(params, mods, d, i, g, star_number, nsteps, t):
    """
    Run the MCMC for a given set of parameters.
    """
    p0, args = assign_args(params, mods, d, i, g, star_number)
    assert np.isfinite(lnprob(p0, *args))
    return MH(p0, nsteps, t, *args)


def gibbs_control(par, mods, d, nsteps, niter, t):
    """
    This function tells the metropolis hastings what parameters to sample in.
    params:
    ------
    par: (list)
        The parameters.
    mods: (list)
        The list of starmodel objects.
    d: (pandas.dataframe)
        The data.
    nsteps: (int)
        Number of samples.
    niter: (int)
        The number of gibbs cycles to perform.
    t: (float)
        The std of the proposal distribution.
    returns:
    -------
    samples: (np.array)
        2d array of samples.
    """
    nstars, ndim = len(d.age.values), len(par)
    n_parameter_sets = nstars + 2  # Num conditionally independent param sets

    # Final sample array
    all_samples = np.zeros((nsteps * n_parameter_sets * niter, ndim))

    # Iterate over niter cycles of parameter sets.
    _set = 0
    for i in range(niter):
        print("set", _set)
        # First sample all the parameters.
        samples, last_sample = run_MCMC(par, mods, d, True, True, None,
                                        nsteps, t)
        all_samples[nsteps*_set:nsteps*(_set+1):] = samples

        # Then sample the gyro parameters only.
        _set += 1
        print("set", _set)
        gyro_samples, last_gyro_sample = run_MCMC(last_sample, mods, d,
                                                  False, True, None, nsteps,
                                                  t)
        all_samples[nsteps*_set:(_set+1)*nsteps, :3] = gyro_samples[:, :3]
        all_samples[nsteps*_set:(_set+1)*nsteps, 3+nstars:3+2*nstars] = \
            gyro_samples[:, 3:]

        # Replace parameter array with the last sample from gyro.
        last_sample[:3] = last_gyro_sample[:3]
        last_sample[3+nstars:3+2*nstars] = last_gyro_sample[3:]

        # Then sample the stars, one by one.
        single_last_samps = []
        _set += 1
        print("set", _set)
        for i in range(nstars):
            samps, last_samp = run_MCMC(last_sample, mods, d, True, False, i,
                                        nsteps, t)
            single_last_samps.append(last_samp)
            all_samples[_set*nsteps:(_set+1)*nsteps, 3+i::nstars] = samps
            _set += 1
            print("set", _set)

        # Replace parameter array with the last sample from single star iso.
        for i in range(nstars):
            last_sample[3+i::nstars] = single_last_samps[i]
        print("last_sample = ", last_sample)

    return all_samples


if __name__ == "__main__":

    DATA_DIR = "/Users/ruthangus/projects/chronometer/chronometer/data"
    d = pd.read_csv(os.path.join(DATA_DIR, "data_file.csv"))

    # Generate the initial parameter array and the mods objects from the data.
    params, mods = pars_and_mods(DATA_DIR)
    params_init = params*1

    start = time.time()  # timeit

    nsteps, niter, t = 50000, 20, 1e-2
    flat = gibbs_control(params, mods, d, nsteps, niter, t)

    end = time.time()
    print("Time taken = ", (end - start)/60, "minutes")

    print("Making corner plot")
    ndim = len(params)
    fig = corner.corner(flat)
    fig.savefig("corner_gibbs_for_realz")

    print("Plotting results and traces")
    # plot_gyro_result(flat, params_init, i, g)

    # # Plot probability
    # plt.clf()
    # plt.plot(sampler.lnprobability.T, "k")
    # plt.savefig("prob_trace")

    # # Plot chains
    # for i in range(ndim):
    #     plt.clf()
    #     plt.plot(sampler.chain[:, :,  i].T, alpha=.5)
    #     plt.savefig("{}_trace".format(i))
