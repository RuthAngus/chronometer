"""
Now use Gibbs sampling to update individual star parameters and global gyro
parameters.
"""

import os

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


def lnlike(params, *args):
    """
    Probability of age and model parameters given rotation period and colour.
    parameters:
    Parameters to pass to the models are selected using the par_inds list in
    *args.
    Whether an "all parameter", "gyro parameter only" or "single star
    isochrone parameters only" gibbs step is determined according to the what
    the elements of par_inds look like.
    This is probably a bad idea.
    ----------
    params: (array)
        The array of log parameters: a, b, n, age.
    args = mods, periods, period_errs, bvs, bv_errs, par_inds
    par_inds: (array)
        The parameters to vary.
    """
    mods, period, period_errs, bv, bv_errs, par_inds = args
    N = len(period)
    gyro_lnlike, iso_lnlike = 0, 0
    if par_inds[0] == 0 and par_inds[1] == 1 and par_inds[2] == 2: # if gyro.

        # If just gyro.
        if len(par_inds) == 3 + N:
            gyro_lnlike = sum(-.5*((period - gyro_model(params, bv))
                                    /period_errs)**2)

        # If gyro and all stars.
        elif len(par_inds) == 3 + 5*N:
            g_par_inds = np.concatenate((par_inds[:3], par_inds[3+N:3+2*N]))
            gyro_lnlike =  sum(-.5*((period - gyro_model(params[g_par_inds],
                                                bv))/period_errs)**2)
            p = params*1
            p[3:3+N] = np.exp(p[3:3+N])
            p[3+N:3+2*N] = np.log10(1e9*np.exp(p[3+N:3+2*N]))
            p[3+3*N:3+4*N] = np.exp(p[3+3*N:3+4*N])
            iso_lnlike = sum([mods[i].lnlike(p[3+i::N]) for i in
                              range(len(mods))])

    # If not gyro but single stars
    else:
        mod_inds = par_inds[0] - 3
        ms = mods[mod_inds]
        p = params*1
        p[0] = np.exp(p[0])
        p[1] = np.log10(1e9*np.exp(p[1]))
        p[3] = np.exp(p[3])
        iso_lnlike = ms.lnlike(p)
    return gyro_lnlike + iso_lnlike


def emcee_lnprob(params, *args):
    """
    The lnprob function used for emcee sampling. Does not break parameters
    into sets.
    """
    mods, period, period_errs, bv, bv_errs, _ = args
    g_par_inds = np.concatenate((par_inds[:3], par_inds[3+N:3+2*N]))
    gyro_lnlike =  sum(-.5*((period - gyro_model(params[g_par_inds], bv))
                            /period_errs)**2)
    p = params*1
    p[3:3+N] = np.exp(p[3:3+N])
    p[3+N:3+2*N] = np.log10(1e9*np.exp(p[3+N:3+2*N]))
    p[3+3*N:3+4*N] = np.exp(p[3+3*N:3+4*N])
    iso_lnlike = sum([mods[i].lnlike(p[3+i::N]) for i in
                        range(len(mods))])
    return gyro_lnlike + iso_lnlike + lnprior(params)


def lnprior(params):
    """
    lnprior on all parameters.
    """
    N = int(len(params - 3)/5)  # Number of stars
    age_prior = sum([np.log(priors.age_prior(np.log10(1e9*np.exp(i))))
                     for i in params[3+N:3+2*N]])
    feh_prior = sum([np.log(priors.feh_prior(i)) for i in
                     params[3+2*N:3+3*N]])
    distance_prior = sum([np.log(priors.distance_prior(np.exp(i))) for i
                          in params[3+3*N:3+4*N]])
    g_prior = priors.lng_prior(params[:3])
    m = (-20 < params) * (params < 20)  # Broad bounds on all params.
    Av = params[3+4*N:3+5*N]
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
    args: (list)
        args to pass to lnlike, including mods, a list of pre-computed star
        model objects.
    """
    return lnlike(params, *args) + lnprior(params)


def MH(par, lnprob, nsteps, t, *args):
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
    args: (list)
    A list of args to pass to the lnlike function.
    mods, periods, period_errs, bvs, bv_errs, par_ind_list = args
    returns:
    --------
    samples: (2d array)
        The posterior samples for a single gibbs iteration.
    par: (array)
        The list of final parameters.
    probs: (array)
        The lnprob chain.
    """
    par_inds = args[-1]
    samples = np.zeros((nsteps, len(par[par_inds])))

    accept, probs = 0, []
    for i in range(nsteps):
        par[par_inds], new_prob, acc = MH_step(par[par_inds], lnprob,
                                               t, *args)
        accept += acc
        probs.append(new_prob)
        samples[i, :] = par[par_inds]
    print("Acceptance fraction = ", accept/float(nsteps))
    return samples, par, probs


def MH_step(par, lnprob, t, *args, emc=False):
    """
    A single Metropolis Hastings step.
    if emc = True, the step is an emcee step instead.
    emcee is run for 10 steps with 64 walkers and the final position is taken
    as the step.
    This is ridiculous but it should demonstrate that tuning is the problem.
    """
    if emc:
        nwalkers, ndim = 64, len(par)
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


def gibbs_control(par, lnprob, nsteps, niter, t, par_inds_list, args):
    """
    This function tells the metropolis hastings what parameters to sample in.
    params:
    ------
    par: (list)
        The parameters.
    lnprob: (function)
        The lnprob function.
    nsteps: (int)
        Number of samples.
    niter: (int)
        The number of gibbs cycles to perform.
    t: (float)
        The covariance matrix of the proposal distribution.
    par_inds_list: (list)
        A list of lists of parameter indices, determining the parameters that
        will be varied during the sampling.
    args: (list)
        A list of the args to parse to lnlike.
        mods, period, period_errs, bv, bv_errs, par_inds = args
    returns:
    -------
    samples: (np.array)
        2d array of samples. (nsteps, ndim)
    lnprobs: (np.array)
        Array of lnprobs
    """

    ndim = len(par)
    n_parameter_sets = len(par_inds_list)

    # Final sample array
    all_samples = np.zeros((nsteps * 2 * niter, ndim))

    # assert len(par_inds_list[0]) == len(par), "You should sample all the " \
        # "parameters first!"

    # Iterate over niter cycles of parameter sets.
    probs = []
    for i in range(niter):  # Loop over Gibbs repeats.
        print("Gibbs iteration ", i, "of ", niter)
        print(par)
        for k in range(len(par_inds_list)):  # loop over parameter sets.
            args[-1] = par_inds_list[k]
            samples, par, pb = MH(par, lnprob, nsteps, t[k], *args)
            all_samples[nsteps*i:nsteps*(i+1), par_inds_list[k]] = samples
            # if len(par_inds_list[k]) == len(par):  # If sampling all params:
            #     all_samples[nsteps*i*2:nsteps*((i*2)+1),
            #                 par_inds_list[k]] = samples
            # else:  # if sampling (non-overlapping) parameter subsets:
            #     all_samples[nsteps*((i*2)+1):nsteps*((i*2)+2),
            #                 par_inds_list[k]] = samples
            probs.append(pb)


    lnprobs = np.array([i for j in probs for i in j])
    return all_samples, lnprobs


def estimate_covariance():
    """
    Return the covariance matrix of the emcee samples.
    """
    with h5py.File("emcee_posterior_samples.h5", "r") as f:
        samples = f["samples"][...]
    return np.cov(samples, rowvar=False)


if __name__ == "__main__":

    # Use Metropolis hastings or emcee?
    run_MH = True
    RESULTS_DIR = "/Users/ruthangus/projects/chronometer/chronometer/MH"
    # RESULTS_DIR = "/Users/ruthangus/projects/chronometer/chronometer/emc"

    # Load the data for the initial parameter array.
    DATA_DIR = "/Users/ruthangus/projects/chronometer/chronometer/data"
    d = pd.read_csv(os.path.join(DATA_DIR, "data_file.csv"))

    # Generate the initial parameter array and the mods objects from the data.
    params, mods = pars_and_mods(DATA_DIR)

    start = time.time()  # timeit

    nsteps, niter = 10000, 5

    # Construct parameter indices for the different parameter sets.
    par_inds = np.arange(len(params))  # All
    N = len(mods)
    gyro_par_inds = np.concatenate((par_inds[:3], par_inds[3+N:3+2*N])) # Gyro
    # par_inds_list = [par_inds, gyro_par_inds]
    par_inds_list = [gyro_par_inds]
    for i in range(N):
        par_inds_list.append(par_inds[3+i::N])  # Iso stars.

    # Create the covariance matrices.
    t = .1*estimate_covariance()
    ts = []
    for i, par_ind in enumerate(par_inds_list):
        ti = np.zeros((len(par_ind), len(par_ind)))
        for j, ind in enumerate(par_ind):
            ti[j] = t[ind][par_ind]
        ts.append(ti)

    # Sample posteriors using either MH gibbs or emcee
    if run_MH:
        args = [mods, d.period.values, d.period_err.values, d.bv.values,
                d.bv_err.values, par_inds_list]
        flat, lnprobs = gibbs_control(params, lnprob, nsteps, niter, ts,
                                    par_inds_list, args)

        # Throw away _number_ Gibbs iterations as burn in.
        number = 1
        burnin = nsteps * number * 2
        flat = flat[burnin:, :]

    else:
        emcee_args = [mods, d.period.values, d.period_err.values, d.bv.values,
                    d.bv_err.values, par_inds_list[0]]
        nwalkers, nsteps, ndim = 64, 10000, len(params)
        p0 = [1e-4*np.random.rand(ndim) + params for i in range(nwalkers)]
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                        args=emcee_args)
        print("burning in...")
        pos, _, _ = sampler.run_mcmc(p0, 3000)
        sampler.reset()
        print("production run...")
        sampler.run_mcmc(pos, nsteps)
        flat = np.reshape(sampler.chain, (nwalkers*nsteps, ndim))

        f = h5py.File("emcee_posterior_samples.h5", "w")
        data = f.create_dataset("samples", np.shape(flat))
        data[:, :] = flat
        f.close()

    end = time.time()
    print("Time taken = ", (end - start)/60, "minutes")

    print("Plotting results and traces")
    plt.clf()
    if run_MH:
        plt.plot(lnprobs)
    else:
        plt.plot(sampler.lnprobability.T)
    plt.xlabel("Time")
    plt.ylabel("ln (probability)")
    plt.savefig(os.path.join(RESULTS_DIR, "prob_trace"))

    print("Making corner plot")
    truths = [.7725, .601, .5189, np.log(1), None, None, np.log(4.56),
            np.log(2.5), np.log(2.5), 0., None, None, np.log(10),
            np.log(2400), np.log(2400), 0., None, None]
    labels = ["$a$", "$b$", "$n$", "$\ln(Mass_1)$", "$\ln(Mass_2)$",
                "$\ln(Mass_3)$", "$\ln(Age_1)$", "$\ln(Age_2)$",
                "$\ln(Age_3)$", "$[Fe/H]_1$", "$[Fe/H]_2$", "$[Fe/H]_3$",
                "$\ln(D_1)$", "$\ln(D_2)$", "$\ln(D_3)$", "$A_{v1}$",
                "$A_{v2}$", "$A_{v3}$"]
    fig = corner.corner(flat, truths=truths, labels=labels)
    fig.savefig(os.path.join(RESULTS_DIR, "demo_corner_gibbs"))

    # Plot chains
    ndim = len(params)
    for i in range(ndim):
        plt.clf()
        if run_MH:
            plt.plot(flat[:, i].T, alpha=.5)
        else:
            plt.plot(sampler.chain[:, :,  i].T, alpha=.5)
        plt.savefig(os.path.join(RESULTS_DIR, "{}_trace".format(i)))
