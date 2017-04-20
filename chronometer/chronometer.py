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
    N = int((len(params) - 3)/5.)
    print(N, "N")
    gyro_lnlike, iso_lnlike = 0, 0
    if par_inds[0] == 0 and par_inds[1] == 1 and par_inds[2] == 2: # if gyro.
        if len(par_inds) == 3 + N:  # If just gyro.
            gyro_lnlike =  sum(-.5*((period - gyro_model(params[par_inds],
                                                bv))/period_errs)**2)
        elif len(par_inds) == 3 + 5*N:  # If gyro and all stars.
            g_par_inds = np.concatenate((par_inds[:3], par_inds[3+N:3+2*N]))
            gyro_lnlike =  sum(-.5*((period - gyro_model(params[g_par_inds],
                                                bv))/period_errs)**2)
            p = params*1
            p[3:3+N] = np.exp(p[:N])
            p[3+N:3+2*N] = np.log10(1e9*np.exp(p[N:2*N]))
            p[3+3*N:3+4*N] = np.exp(p[3*N:4*N])
            iso_lnlike = [mods[i].lnlike(params[par_inds][3+i::N]) for i in
                          range(len((mods)))]
        else:
            assert len(par_inds) - 3 - N == 0  # Not gyro + single stars.
    else:  # If not gyro but single stars
        print("yes")
        mod_inds = par_inds[0] - 3
        print(mod_inds)
        ms = mods[mod_inds]
        print(ms)
        # try:
        #     if len(ms) > 1:
        #         print(params[par_inds][i::N])
        #         iso_lnlike = [ms[i].lnlike(params[par_inds][i::N]) for i in
        #                     range(len((ms)))]
        # except TypeError:
        print(params[par_inds])
        iso_lnlike = ms.lnlike(params[par_inds])
    return gyro_lnlike + iso_lnlike


def lnprior(params):
    """
    lnprior on the parameters when both iso and gyro parameters are being
    inferred.
    """
    Nstars = int(len(params - 3)/5)

    age_prior = sum([np.log(priors.age_prior(np.log10(1e9*np.exp(i))))
                     for i in params[N:2*N]])
    feh_prior = sum([np.log(priors.feh_prior(i)) for i in params[2*N:3*N]])
    distance_prior = sum([np.log(priors.distance_prior(np.exp(i))) for i
                          in params[3*N:4*N]])
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
    samples = np.zeros((nsteps, len(par[par_inds])))
    accept, probs = 0, []
    for i in range(nsteps):
        par[par_inds], new_prob, acc = MH_step(par[par_inds], lnprob, t,
                                               *args)
        accept += acc
        probs.append(new_prob)
        samples[i, :] = par[par_inds]
    print("Acceptance fraction = ", accept/float(nsteps))
    return samples, par, probs


def MH_step(par, lnprob, t, *args):
    newp = par + np.random.randn(len(par))*t
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

    N, ndim = len(d.age.values), len(par)
    n_parameter_sets = N + 2  # Num conditionally independent param sets

    # Final sample array
    all_samples = np.zeros((nsteps * n_parameter_sets * niter, ndim))

    par_inds = np.arange(ndim)
    gyro_par_inds = np.concatenate((par_inds[:3], par_inds[3+N:3+2*N]))
    iso_par_inds = []
    for i in range(N):
        iso_par_inds.append(par_inds[3+i::N])

    args = [mods, d.period.values, d.period_err.values, d.bv.values,
            d.bv_err.values,
            par_inds]

    # Iterate over niter cycles of parameter sets.
    _set, probs = 0, []
    for i in range(niter):
        # First sample all the parameters.
        args[-1] = par_inds
        samples, par, pb = MH(par, lnprob, nsteps, t, *args)
        all_samples[nsteps*_set:nsteps*(_set+1), :][par_inds] = samples
        probs.append(pb)

        # Then sample the gyro parameters only.
        _set += 1
        args[-1] = gyro_par_inds
        gyro_samples, last_gyro_sample, pb = run_MCMC(par, lnprob, nsteps, t,
                                                      *args)
        all_samples[nsteps*_set:(_set+1)*nsteps, :][gyro_par_inds] = \
            gyro_samples
        probs.append(pb)

        # Replace parameter array with the last sample from gyro.
        par[gyro_par_inds] = last_gyro_sample

        # Then sample the stars, one by one.
        single_last_samps = []
        _set += 1
        for i in range(nstars):
            args[-1] = iso_par_inds[i]
            samps, last_samp, pb = run_MCMC(par, lnprob, nsteps, t, *args)
            all_samples[_set*nsteps:(_set+1)*nsteps, :][iso_par_inds[i]] = \
                samps
            probs.append(pb)
            _set += 1
            par[iso_par_inds[i]] = last_samp

    lnprobs = np.array([i for j in probs for i in j])
    return all_samples, lnprobs


if __name__ == "__main__":

    DATA_DIR = "/Users/ruthangus/projects/chronometer/chronometer/data"
    d = pd.read_csv(os.path.join(DATA_DIR, "data_file.csv"))

    # Generate the initial parameter array and the mods objects from the data.
    params, mods = pars_and_mods(DATA_DIR)

    start = time.time()  # timeit

    t = [.01, .01, .01, .03, .1, .1, .3, .3, .3, .1, .2, .2, .02, .2, .2, .01,
         .2, .2]
    nsteps, niter = 1000, 3
    flat, lnprobs = gibbs_control(params, mods, d, nsteps, niter, t)

    end = time.time()
    print("Time taken = ", (end - start)/60, "minutes")

    print("Making corner plot")
    ndim = len(params)
    fig = corner.corner(flat)
    fig.savefig("corner_gibbs_for_realz")

    print("Plotting results and traces")
    plt.clf()
    plt.plot(lnprobs)
    plt.xlabel("Time")
    plt.ylabel("ln (probability)")
    plt.savefig("prob_trace")
