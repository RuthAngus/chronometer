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
from models import gc_model, gyro_model, action_age
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
    N, ngyro, nglob, nind = get_n_things(args[0], params)
    mods, period, period_errs, bv, bv_errs, Jz, Jz_err, par_inds = args
    age_par_inds = par_inds[nglob+N:nglob+2*N]
    g_par_inds = np.concatenate((par_inds[:ngyro],
                                 par_inds[nglob+N:nglob+2*N]))
    m = (bv > .4) * (np.isfinite(period))
    g_par_inds_mask = np.concatenate((g_par_inds[:3], g_par_inds[3:][m]))
    gyro_lnlike = sum(-.5*((period[m] - gyro_model(params[g_par_inds_mask],
                                                     bv[m]))
                            /period_errs[m])**2)

    # A list of kinematic indices.
    kin_inds = list(range(nglob+N, nglob+N+nind))
    kin_inds.insert(0, ngyro)
    kin_lnlike = action_age(params[kin_inds], Jz, Jz_err)

    p = params*1
    p[nglob:nglob+N] = np.exp(p[nglob:nglob+N])
    p[nglob+N:nglob+2*N] = np.log10(1e9*np.exp(p[nglob+N:nglob+2*N]))
    p[nglob+nglob*N:nglob+4*N] = np.exp(p[nglob+nglob*N:nglob+4*N])
    iso_lnlike = sum([mods[i].lnlike(p[nglob+i::N]) for i in
                      range(len(mods))])
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
    d = pd.read_csv(os.path.join(DATA_DIR, "action_data.csv"))

    # Generate the initial parameter array and the mods objects from the data
    global_params = np.array([.7725, .601, .5189, np.log(350.)])  # a b n beta

    params, mods = pars_and_mods(d, global_params)

    # Set nsteps and niter.
    N, ngyro, nglob, nind = get_n_things(mods, params)
    print(N, "stars")

    # Construct parameter indices for the different parameter sets.
    par_inds = np.arange(len(params))  # All

    emcee_args = [mods, d.prot.values,
                d.prot_err.values, d.bv.values,
                d.bv_err.values, d.Jz, d.Jz_err, par_inds]
    nwalkers, nsteps, ndim = 64, 15000, len(params)
    p0 = [1e-4*np.random.rand(ndim) + params for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, emcee_lnprob,
                                    args=emcee_args)
    print("burning in...")
    pos, _, _ = sampler.run_mcmc(p0, 10000)
    sampler.reset()
    print("production run...")
    sampler.run_mcmc(pos, nsteps)
    flat = np.reshape(sampler.chain, (nwalkers*nsteps, ndim))

    f = h5py.File("emcee_posterior_samples.h5", "w")
    data = f.create_dataset("action_samples", np.shape(flat))
    data[:, :] = flat
    f.close()

    emcee_labels = ["$a$", "$b$", "$n$", "$\\beta$",
                    "$\ln(Mass_1)$", "$\ln(Mass_2)$", "$\ln(Mass_3)$",
                    "$\ln(Mass_4)$", "$\ln(Mass_5)$",
                    "$\ln(Age_{1,i})$", "$\ln(Age_{2,i})$",
                    "$\ln(Age_{3,i})$", "$\ln(Age_{4,i})$",
                    "$\ln(Age_{5,i})$",
                    "$[Fe/H]_1$", "$[Fe/H]_2$", "$[Fe/H]_3$",
                    "$[Fe/H]_4$", "$[Fe/H]_5$",
                    "$\ln(D_1)$", "$\ln(D_2)$", "$\ln(D_3)$",
                    "$\ln(D_4)$", "$\ln(D_5)$",
                    "$A_{v1}$", "$A_{v2}$", "$A_{v3}$",
                    "$A_{v4}$", "$A_{v5}$"]
    fig = corner.corner(flat, labels=emcee_labels)
    fig.savefig(os.path.join(RESULTS_DIR, "emcee_corner"))

    # Plot chains
    for i in range(ndim):
        plt.clf()
        plt.plot(sampler.chain[:, :,  i].T, alpha=.5)
        plt.savefig(os.path.join(RESULTS_DIR, "{}_trace".format(i)))
