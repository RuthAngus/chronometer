"""
A version of chronometer where we step through each parameter separately.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import corner
import time

from chronometer import get_n_things, estimate_covariance, augment
from utils import pars_and_mods
from models import gyro_model, action_age, action_age_MH
import priors


def lnprob(params, *args):
    """
    The lnprob function used for emcee sampling. Does not break parameters
    into sets.
    """
    mods, period, period_errs, bv, bv_errs, Jz, Jz_err = args
    N, ngyro, nglob, nind = get_n_things(args[0], params)

    mp = period > 0.
    mj = Jz > 0.
    g_par_inds = np.concatenate((np.array([0, 1, 2]),
                                 np.arange(nglob+N, nglob+2*N)))
    k_par_inds = np.concatenate((np.array([3]), np.arange(nglob+N,
                                                          nglob+2*N)))

    gmask = np.concatenate((g_par_inds[:3], g_par_inds[3:][mp]))
    kmask = np.concatenate((k_par_inds[:1], k_par_inds[1:][mj]))
    gyro_lnlike = sum(-.5*((period[mp] - gyro_model(params[gmask],
                                                    bv[mp]))
                            /period_errs[mp])**2)

    kin_lnlike = action_age_MH(params[kmask], Jz[mj], Jz_err[mj])

    p = params*1
    p[nglob:nglob+N] = np.exp(p[nglob:nglob+N])  # mass
    p[nglob+N:nglob+2*N] = np.log10(1e9*np.exp(p[nglob+N:nglob+2*N]))  # age
    p[nglob+3*N:nglob+4*N] = np.exp(p[nglob+3*N:nglob+4*N])  # dist
    iso_lnlike = sum([mods[i].lnlike(p[nglob+i::N]) for i in
                      range(len(mods))])

    # gyro_lnlike = 0
    # kin_lnlike = 0
    # iso_lnlike = 0
    return gyro_lnlike + iso_lnlike + kin_lnlike + lnprior(params, *args)


def lnprior(params, *args):
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


def MH(par, lnprob, nsteps, niter, t, *args):
    """
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
    samples = np.zeros((nsteps*len(par)*niter, len(par)))

    labels = ["$a$", "$b$", "$n$", "$\\beta$",
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

    accept, probs = 0, []
    accept_list = [[] for i in range(len(par))]
    for k in range(niter):  # loop over iterations
        if (k + 1) % 100 == 0:
            print("iteration", k + 1, "of", niter)
            print("Acceptance fraction = ", accept/(k*nsteps*len(par)))
        for j in range(len(par)):  # loop over parameters
            for i in range(nsteps):  # Do the MCMC
                par, new_prob, acc = MH_step(par, lnprob, j, t, *args)
                accept += acc
                accept_list[j].append(acc)
                probs.append(new_prob)
                samples[i + j*nsteps + k*len(par)*nsteps, :] = par

    for i, l in enumerate(accept_list):
        print("param", i, sum(l)/(niter*nsteps))
    return samples, par, probs


def MH_step(par, lnprob, i, t, *args, emc=False):
    """
    A single Metropolis step.
    if emc = True, the step is an emcee step instead.
    emcee is run for 10 steps with 64 walkers and the final position is taken
    as the step.
    This is ridiculous but it should demonstrate that tuning is the problem.
    """
    newp = par*1
    # t = np.ones_like(t) * .01  # FIXME
    newp[i] = (par + np.random.multivariate_normal(np.zeros((len(par))),
                                                   t))[i]
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
    RESULTS_DIR = "/Users/ruthangus/projects/chronometer/chronometer/MH"

    # Load the data for the initial parameter array.
    DATA_DIR = "/Users/ruthangus/projects/chronometer/chronometer/data"
    # d = pd.read_csv(os.path.join(DATA_DIR, "action_data.csv"))
    d = pd.read_csv(os.path.join(DATA_DIR, "fake_data.csv"))
    d = d.iloc[:6]

    # Generate the initial parameter array and the mods objects from the data
    global_params = np.array([.7725, .601, .5189, np.log(350.)])  # a b n beta

    params, mods = pars_and_mods(d, global_params)

    # Set nsteps and niter.
    nsteps = 1
    niter = 10000
    N, ngyro, nglob, nind = get_n_things(mods, params)
    print(N, "stars")

    # Create the covariance matrix.
    t = estimate_covariance(N, "emcee_posterior_samples_0525.h5")
    t = augment(t, N - 5, 5)

    start = time.time()

    # Sample posteriors using MH gibbs
    args = [mods, d.prot.values, d.prot_err.values, d.bv.values,
            d.bv_err.values, d.Jz.values, d.Jz_err.values]

    # Test lnprob
    print("params = ", params)
    print("lnprob = ", lnprob(params, *args))
    flat, par, probs = MH(params, lnprob, nsteps, niter, t, *args)

    end = time.time()
    print("Time taken = ", (end - start)/60, "minutes")

    ml = ["$\ln(Mass_{})$".format(i) for i in range(N)]
    al = ["$\ln(Age_{})$".format(i) for i in range(N)]
    fl = ["$[Fe/H]_{}$".format(i) for i in range(N)]
    dl = ["$\ln(D_{})$".format(i) for i in range(N)]
    avl = ["$A_v{}$".format(i) for i in range(N)]
    labels = ["$a$", "$b$", "$n$", "$\\beta$"] + ml + al + fl + dl + avl

    tr = pd.read_csv("truths.txt")
    truths = np.concatenate((np.array(global_params),
                             np.log(tr.mass.values[:N]),
                             np.log(tr.age.values[:N]), tr.feh.values[:N],
                             np.log(tr.distance.values[:N]), tr.Av.values[:N]))

    print("Making corner plot")
    fig = corner.corner(flat, truths=truths, labels=labels)
    fig.savefig(os.path.join(RESULTS_DIR, "simple_corner_gibbs"))
