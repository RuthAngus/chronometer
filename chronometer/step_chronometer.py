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

import corner
import priors
from models import gc_model, gyro_model

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


def lnlike(params, *args):
    """
    Probability of age and model parameters given rotation period and colour.
    parameters:
    ----------
    params: (array)
        The array of log parameters: a, b, n, age.
    par_inds: (array)
        The parameters to vary.
    """
    mods, period, period_errs, bv, bv_errs, par_inds = args
    N = len(mods)
    g_par_inds = np.concatenate((np.arange(3), np.arange(3+N, 3+2*N)))
    gyro_lnlike =  sum(-.5*((period - gyro_model(params[g_par_inds],
                                        bv))/period_errs)**2)
    p = params*1
    p[3:3+N] = np.exp(p[3:3+N])
    p[3+N:3+2*N] = np.log10(1e9*np.exp(p[3+N:3+2*N]))
    p[3+3*N:3+4*N] = np.exp(p[3+3*N:3+4*N])
    iso_lnlike = sum([mods[i].lnlike(p[3+i::N]) for i in range(len(mods))])
    return gyro_lnlike + iso_lnlike


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
    mod: (list)
        list of pre-computed star model objects.
    """
    return lnlike(params, *args) + lnprior(params)


def MH(par, lnprob, nsteps, t, *args):
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
    par_inds = args[-1]
    samples = np.zeros((nsteps, len(par)))
    accept, probs = 0, []
    for i in range(nsteps):
        par, new_prob, acc = MH_step(par, lnprob, t, *args)
        accept += acc
        probs.append(new_prob)
        samples[i, :] = par
    print(accept)
    print(nsteps)
    print("Acceptance fraction = ", accept/float(nsteps))
    return samples, par, probs


def MH_step(par, lnprob, t, *args):
    newp = par*1
    par_inds = args[-1]
    newp[par_inds] += np.random.randn(len(par[par_inds]))*t[par_inds]
    new_lnprob = lnprob(newp, *args)
    alpha = np.exp(new_lnprob)/np.exp(lnprob(par, *args))
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
    return par, new_lnprob, accept


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
        2d array of samples. (nsteps, ndim)
    lnprobs: (np.array)
        Array of lnprobs
    """

    N, ndim = len(d.age.values), len(par)
    n_parameter_sets = N + 1 + 1  # Number of param sets, including all.

    # Final sample array
    all_samples = np.zeros((nsteps * n_parameter_sets * niter, ndim))

    # Construct parameter indices for the different parameter sets.
    par_inds = np.arange(ndim)  # All
    gyro_par_inds = np.concatenate((par_inds[:3], par_inds[3+N:3+2*N])) # Gyro
    iso_par_inds = []
    for i in range(N):
        iso_par_inds.append(par_inds[3+i::N])  # Iso stars.

    args = [mods, d.period.values, d.period_err.values, d.bv.values,
            d.bv_err.values,
            par_inds]

    # Iterate over niter cycles of parameter sets.
    probs = []
    for i in range(niter):
        print("all")
        # First sample all the parameters.
        samples, par, pb = MH(par, lnprob, nsteps, t, *args)
        all_samples[nsteps*i:nsteps*(i+1), :] = samples
        probs.append(pb)

        # Then sample the gyro parameters only.
        print("gyro")
        args[-1] = gyro_par_inds
        gyro_samples, par, pb = MH(par, lnprob, nsteps, t, *args)
        all_samples[nsteps*i:(i+1)*nsteps, :] = gyro_samples
        probs.append(pb)

        # Replace parameter array with the last sample from gyro.
        # par[gyro_par_inds] = last_gyro_sample

        # Then sample the stars, one by one.
        single_last_samps = []
        for j in range(N):
            print("iso", j)
            args[-1] = iso_par_inds[j]
            samps, par, pb = MH(par, lnprob, nsteps, t, *args)
            all_samples[i*nsteps:(i+1)*nsteps, :] = samps
            probs.append(pb)
            # par[iso_par_inds[i]] = last_samp

    lnprobs = np.array([i for j in probs for i in j])
    return all_samples, lnprobs


if __name__ == "__main__":

    DATA_DIR = "/Users/ruthangus/projects/chronometer/chronometer/data"
    d = pd.read_csv(os.path.join(DATA_DIR, "data_file.csv"))

    # Generate the initial parameter array and the mods objects from the data.
    print("Instantiating starmodels")
    params, mods = pars_and_mods(DATA_DIR)

    start = time.time()  # timeit

    t = .1*np.array([.01, .01, .01, .03, .1, .1, .3, .3, .3, .1, .2, .2, .02,
                     .2, .2, .01, .2, .2])
    nsteps, niter = 10000, 3
    flat, lnprobs = gibbs_control(params, mods, d, nsteps, niter, t)

    end = time.time()
    print("Time taken = ", (end - start)/60, "minutes")

    print("Plotting results and traces")
    plt.clf()
    plt.plot(lnprobs)
    plt.xlabel("Time")
    plt.ylabel("ln (probability)")
    plt.savefig("prob_trace")

    print("Making corner plot")
    ndim = len(params)
    fig = corner.corner(flat)
    fig.savefig("corner_gibbs_for_realz")
