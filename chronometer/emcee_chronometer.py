"""
Use emcee to produce posterior samples.
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
from models import gyro_model, action_age, pure_gyro_model, pure_action_age
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


def get_n_things(mods, params):
    """
    Figure out number of stars, number of global params, etc.
    """
    N, nind = len(mods), 5
    ngyro = 3
    nglob = len(params) - (N*nind)
    return N, ngyro, nglob, nind


def emcee_lnprob(params, *args):
    """
    The lnprob function used for emcee sampling. Does not break parameters
    into sets.
    """
    mods, period, period_errs, bv, bv_errs, Jz, Jz_err, N, ngyro, nglob, \
        nind, g_par_inds_mask, kin_inds, m = args

    gyro_lnlike = sum(-.5*((period[m] - gyro_model(params[g_par_inds_mask],
                                                     bv[m]))
                            /period_errs[m])**2)

    mj = Jz > 0.
    kin_lnlike = action_age(params[kin_inds], Jz, Jz_err, mj)

    p = params*1
    p[nglob:nglob+N] = np.exp(p[nglob:nglob+N])  # mass
    p[nglob+N:nglob+2*N] = np.log10(1e9*np.exp(p[nglob+N:nglob+2*N]))  # age
    p[nglob+3*N:nglob+4*N] = np.exp(p[nglob+3*N:nglob+4*N])  # dist
    iso_lnlike = sum([mods[i].lnlike(p[nglob+i::N]) for i in
                      range(len(mods))])

    # iso_lnlike = 0
    # kin_lnlike = 0
    # gyro_lnlike = 0
    return gyro_lnlike + iso_lnlike + kin_lnlike + \
        emcee_lnprior(params, *args)


def emcee_lnprior(params, *args):
    """
    lnprior on all parameters.
    """
    N, ngyro, nglob, nind = get_n_things(args[0], params)
    g_prior = priors.lng_prior(params[:ngyro])
    age_prior = sum([np.log(priors.age_prior(np.log10(1e9*np.exp(i))))
                     for i in params[nglob+N:nglob+2*N]])
    feh_prior = sum(np.log(priors.feh_prior(params[nglob+2*N:nglob+3*N])))
    distance_prior = sum(np.log(priors.distance_prior(
                            np.exp(params[nglob+3*N:nglob+4*N]))))

    mAv = (0 <= params[nglob+4*N:nglob+5*N]) * \
        (params[nglob+4*N:nglob+5*N] < 1)  # Prior on A_v
    m = (-20 < params) * (params < 20)  # Broad bounds on all params.
    mbeta = -20 < params[ngyro] < 20  # Prior on beta

    if sum(m) == len(m) and sum(mAv) == len(mAv) and mbeta:
        return g_prior + age_prior + feh_prior + distance_prior
    else:
        return -np.inf


if __name__ == "__main__":

    RESULTS_DIR = "/Users/ruthangus/projects/chronometer/chronometer/emc"

    # Load the data for the initial parameter array.
    DATA_DIR = "/Users/ruthangus/projects/chronometer/chronometer/data"
    # d = pd.read_csv(os.path.join(DATA_DIR, "action_data.csv"))
    d = pd.read_csv(os.path.join(DATA_DIR, "fake_data.csv"))
    N = 3
    d = d.iloc[:N]

    # Generate the initial parameter array and the mods objects from the data
    global_params = np.array([.7725, .601, .5189, np.log(350.)])  # a b n beta

    params, mods = pars_and_mods(d, global_params)
    print("intial parameters = ", params)

    # Set nsteps and niter.
    N, ngyro, nglob, nind = get_n_things(mods, params)
    print(N, "stars")
    print(ngyro, "gyro parameters")
    print(nglob, "global parameters")
    print(nind, "parameters per star")

    # Construct parameter indices for the different parameter sets.
    par_inds = np.arange(len(params))  # All
    g_par_inds = np.concatenate((par_inds[:ngyro],
                                 par_inds[nglob+N:nglob+2*N]))
    m = (d.bv.values > .4) * (np.isfinite(d.prot.values))
    g_par_inds_mask = np.concatenate((g_par_inds[:3], g_par_inds[3:][m]))

    # A list of kinematic indices.
    kin_inds = list(range(nglob+N, nglob+2*N))
    kin_inds.insert(0, ngyro)

    emcee_args = [mods, d.prot.values, d.prot_err.values, d.bv.values,
                  d.bv_err.values, d.Jz.values, d.Jz_err.values, N, ngyro,
                  nglob, nind, g_par_inds_mask, kin_inds, m]

    # Test lnprob
    print("lnprob = ", emcee_lnprob(params, *emcee_args))

    nwalkers, nsteps, ndim, mult = 2*len(params) + 10, 10000, len(params), 5
    np.random.seed(1234)
    p0 = [1e-4*np.random.rand(ndim) + params for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, emcee_lnprob,
                                    args=emcee_args)
    print("burning in...")
    start = time.time()
    pos, _, _ = sampler.run_mcmc(p0, nsteps*10)
    end = time.time()
    print("Time = ", (end - start)/60, "minutes")
    print("Predicted run time = ", (end - start)/60 * mult, "minutes")
    sampler.reset()
    print("production run...")
    start = time.time()
    sampler.run_mcmc(pos, mult*nsteps)
    end = time.time()
    print("Time = ", (end - start)/60, "minutes")
    flat = np.reshape(sampler.chain, (nwalkers*nsteps*mult, ndim))

    f = h5py.File("emcee_posterior_samples.h5", "w")
    data = f.create_dataset("action_samples", np.shape(flat))
    data[:, :] = flat
    f.close()

    ml = ["$\ln(Mass_{})$".format(i) for i in range(N)]
    al = ["$\ln(Age_{})$".format(i) for i in range(N)]
    fl = ["$[Fe/H]_{}$".format(i) for i in range(N)]
    dl = ["$\ln(D_{})$".format(i) for i in range(N)]
    avl = ["$A_v{}$".format(i) for i in range(N)]
    emcee_labels = ["$a$", "$b$", "$n$", "$\\beta$"] + ml + al + fl + dl + avl

    truths = np.concatenate((np.array(global_params),
                             np.log(d.mass.values[:N]),
                             np.log(d.age.values[:N]), d.feh.values[:N],
                             np.log(d.distance.values[:N]), d.Av.values[:N]))

    print("Making corner plot")
    fig = corner.corner(flat, labels=emcee_labels, truths=truths)
    fig.savefig(os.path.join(RESULTS_DIR, "emcee_corner"))

    # Plot chains
    print("plotting chains")
    for i in range(ndim):
        print(i, "of", len(params))
        plt.clf()
        plt.plot(sampler.chain[:, :,  i].T, alpha=.5)
        plt.savefig(os.path.join(RESULTS_DIR, "{}_trace".format(i)))
