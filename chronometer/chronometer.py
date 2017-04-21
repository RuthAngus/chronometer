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
    par_inds = args[-1]
    samples = np.zeros((nsteps, len(par[par_inds])))
    accept, probs = 0, []
    for i in range(nsteps):
        par[par_inds], new_prob, acc = MH_step(par[par_inds], lnprob,
                                               t[par_inds], *args)
        accept += acc
        probs.append(new_prob)
        samples[i, :] = par[par_inds]
    print("Acceptance fraction = ", accept/float(nsteps))
    return samples, par, probs


def MH_step(par, lnprob, t, *args):
    newp = par + np.random.randn(len(par))*(t *
                                            np.exp(np.random.uniform(-7, 3)))
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
    return par, new_lnprob, accept


def gibbs_control(par, lnprob, nsteps, niter, t, par_inds_list, args):
    """
    This function tells the metropolis hastings what parameters to sample in.
    params:
    ------
    par: (list)
        The parameters.
    nsteps: (int)
        Number of samples.
    niter: (int)
        The number of gibbs cycles to perform.
    t: (float)
        The std of the proposal distribution.
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
    skip = 2
    if len(par_inds_list[0]) == 1:
        skip = 1
    all_samples = np.zeros((nsteps * skip * niter, ndim))

    assert len(par_inds_list[0]) == len(par), "You should sample all the " \
        "parameters first!"

    # Iterate over niter cycles of parameter sets.
    probs = []
    for i in range(niter):  # Loop over Gibbs repeats.
        print("Gibbs iteration ", i, "of ", niter)
        for k in range(len(par_inds_list)):  # loop over parameter sets.
            print("Parameter set ", k, "of", len(par_inds_list))

            args[-1] = par_inds_list[k]
            samples, par, pb = MH(par, lnprob, nsteps, t, *args)
            if len(par_inds_list[k]) == len(par):  # If sampling all params:
                all_samples[nsteps*i*skip:nsteps*((i*skip)+1),
                            par_inds_list[k]] = samples
            else:  # if sampling (non-overlapping) parameter subsets:
                all_samples[nsteps*((i*2)+1):nsteps*((i*2)+2),
                            par_inds_list[k]] = samples
            probs.append(pb)

    lnprobs = np.array([i for j in probs for i in j])
    return all_samples, lnprobs


if __name__ == "__main__":

    DATA_DIR = "/Users/ruthangus/projects/chronometer/chronometer/data"
    d = pd.read_csv(os.path.join(DATA_DIR, "data_file.csv"))

    # Generate the initial parameter array and the mods objects from the data.
    params, mods = pars_and_mods(DATA_DIR)

    start = time.time()  # timeit

    t = .1*np.array([.01, .01, .01, .03, .1, .1, .3, .3, .3, .1, .2, .2, .02,
                     .2, .2, .01, .2, .2])
    t = np.ones(len(t))*1e-3
    nsteps, niter = 1000, 3


    # Construct parameter indices for the different parameter sets.
    par_inds = np.arange(ndim)  # All
    gyro_par_inds = np.concatenate((par_inds[:3], par_inds[3+N:3+2*N])) # Gyro
    par_inds_list = [par_inds, gyro_par_inds]
    for i in range(N):
        par_inds_list.append(par_inds[3+i::N])  # Iso stars.
    args = [mods, d.period.values, d.period_err.values, d.bv.values,
            d.bv_err.values, par_inds_list[0]]
    flat, lnprobs = gibbs_control(params, nsteps, niter, t, par_inds_list,
                                  *args)

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
