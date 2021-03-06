"""
A version of chronometer where we step through each parameter separately.
This version uses only the isochronal information.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import corner
import time
import h5py
import sys

from utils import pars_and_mods, get_n_things, estimate_covariance, augment
from models import gyro_model, action_age, action_age_MH
import priors


def lnprob(params, *args):
    """
    The lnprob function used for emcee sampling. Does not break parameters
    into sets.
    """
    mods, period, period_errs, bv, bv_errs, Jz, Jz_err = args
    N, ngyro, nglob, nind = get_n_things(args[0], params)

    p = params*1
    p[nglob:nglob+N] = np.exp(p[nglob:nglob+N])  # mass
    p[nglob+N:nglob+2*N] = np.log10(1e9*np.exp(p[nglob+N:nglob+2*N]))  # age
    p[nglob+3*N:nglob+4*N] = np.exp(p[nglob+3*N:nglob+4*N])  # dist
    iso_lnlike = sum([mods[i].lnlike(p[nglob+i::N]) for i in
                      range(len(mods))])

    return iso_lnlike + lnprior(params, *args)


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

    accept, probs = 0, []
    accept_list = [[] for i in range(len(par))]
    for k in range(niter):  # loop over iterations
        if (k + 1) % 100 == 0:
            print("iteration", k + 1, "of", niter, "iso only")
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


def MH_step(par, lnprob, i, t, *args):
    """
    A single Metropolis step.
    """
    newp = par*1
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


def burnin(params, mods, args, t, niter=10000, nsteps=1, clobber=False):
    """
    Run a single Gibbs chain until it has burnt in.
    params:
    ------
    params: (array)
        The initial parameter array. Should contain:
        [a, b, n, beta, Mass_i, Age_i, Feh_i, Distance_i, Av_i, ...,
        Mass_n, Age_n, Feh_n, Distance_n, Av_n] for stars i to n.
    mods: (list)
        A list of starmodel objects, created using colours and parallax.
    args: (list)
        A list of arguments to pass to the lnprob function.
        Contains rotation periods, B-V colours and vertical actions.
    t: (nd array)
        The covariance matrix created during an emcee run, used for the
        proposal distributions.
    niter: (int)
        The number of Gibbs sets or parameter cycles.
    nsteps: (int)
        The number of steps taken in each metropolis MCMC run.
        Usually set to 1.
    returns:
    -------
    parameters: (array)
        The array of final parameters that come out of burn in.
    """
    fn = os.path.join(RESULTS_DIR, "burnin_results_{}.csv".format(len(mods)))
    if not clobber and os.path.exists(fn):
        df = pd.read_csv(fn)
        par = df.params.values

    else:
        print("Running burn in...")
        start = time.time()
        flat, par, probs = MH(params, lnprob, nsteps, niter, t, *args)
        end = time.time()
        print("Time taken = ", (end - start)/60, "minutes")
        df = pd.DataFrame({"params": par})
        df.to_csv(fn)
    return par


def run_multiple_chains(fn, params, mods, args, t, niter=10000, nsteps=1,
                        plot=True):
    """
    After burn in, start several chains running. To run multiple chains, simply
    run this function multiple times.
    params:
    ------
    fn: (str)
        The name of the .h5 sample file.
    """
    # load burn in results
    df = pd.read_csv(os.path.join(RESULTS_DIR,
                                  "burnin_results_{}.csv".format(len(mods))))
    params = df.params.values

    # Run Gibbs
    print("Running chain {}".format(fn))
    start = time.time()
    flat, par, probs = MH(params, lnprob, nsteps, niter, t, *args)
    end = time.time()
    print("Time taken = ", (end - start)/60, "minutes")

    # Save samples
    f = h5py.File(os.path.join(RESULTS_DIR, "{}.h5".format(fn)), "w")
    data = f.create_dataset("samples", np.shape(flat))
    data[:, :] = flat
    f.close()

    # Make plot
    if plot:
        N, ngyro, nglob, nind = get_n_things(mods, params)
        ml = ["$\ln(Mass_{})$".format(i) for i in range(N)]
        al = ["$\ln(Age_{})$".format(i) for i in range(N)]
        fl = ["$[Fe/H]_{}$".format(i) for i in range(N)]
        dl = ["$\ln(D_{})$".format(i) for i in range(N)]
        avl = ["$A_v{}$".format(i) for i in range(N)]
        labels = ["$a$", "$b$", "$n$", "$\\beta$"] + ml + al + fl + dl + avl
        tr = pd.read_csv("data/fake_data.csv")
        truths = np.concatenate((np.array(global_params),
                                 np.log(tr.mass.values[:N]),
                                 np.log(tr.age.values[:N]), tr.feh.values[:N],
                                 np.log(tr.distance.values[:N]),
                                 tr.Av.values[:N]))
        print("Making corner plot")
        fig = corner.corner(flat, truths=truths, labels=labels)
        fig.savefig(os.path.join(RESULTS_DIR, fn))


def combine_samples(fn_list, fn, mods, params, plot=True):
    """
    Gather up the parallelised results into one set of samples.
    Calculate Gelman & Rubin convergence diagnostic.
    params:
    ------
    fn_list: (list)
        A list of the names of files containing samples you want to join
        (exluding the .h5)
    fn: (str)
        New file name for all the samples.
    """

    # Load first set of samples
    with h5py.File(os.path.join(RESULTS_DIR, "{}.h5".format(fn_list[0])),
                   "r") as f:
        samples = f["samples"][...]

    # Append subsequent sets of samples
    for fl in fn_list[1:]:
        with h5py.File(os.path.join(RESULTS_DIR, "{}.h5".format(fl)),
                       "r") as f:
            s = f["samples"][...]
        samples = np.vstack((samples, s))

    # Save large sample set.
    fs = h5py.File(os.path.join(RESULTS_DIR, "{}.h5".format(fn)), "w")
    data = fs.create_dataset("samples", np.shape(samples))
    data[:, :] = samples
    fs.close()

    # Make plot
    if plot:
        N, ngyro, nglob, nind = get_n_things(mods, params)
        ml = ["$\ln(Mass_{})$".format(i) for i in range(N)]
        al = ["$\ln(Age_{})$".format(i) for i in range(N)]
        fl = ["$[Fe/H]_{}$".format(i) for i in range(N)]
        dl = ["$\ln(D_{})$".format(i) for i in range(N)]
        avl = ["$A_v{}$".format(i) for i in range(N)]
        labels = ["$a$", "$b$", "$n$", "$\\beta$"] + ml + al + fl + dl + avl
        tr = pd.read_csv("data/fake_data.csv")
        truths = np.concatenate((np.array(global_params),
                                 np.log(tr.mass.values[:N]),
                                 np.log(tr.age.values[:N]), tr.feh.values[:N],
                                 np.log(tr.distance.values[:N]),
                                 tr.Av.values[:N]))
        print("Making corner plot")
        fig = corner.corner(samples, truths=truths, labels=labels)
        fig.savefig(os.path.join(RESULTS_DIR, fn))


if __name__ == "__main__":
    RESULTS_DIR = "/Users/ruthangus/projects/chronometer/chronometer/MH"

    # Load the data for the initial parameter array.
    DATA_DIR = "/Users/ruthangus/projects/chronometer/chronometer/data"
    # d = pd.read_csv(os.path.join(DATA_DIR, "action_data.csv"))
    d = pd.read_csv(os.path.join(DATA_DIR, "fake_data.csv"))
    d = d.iloc[:10]

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

    # Sample posteriors using MH gibbs
    args = [mods, d.prot.values, d.prot_err.values, d.bv.values,
            d.bv_err.values, d.Jz.values, d.Jz_err.values]

    # Test lnprob
    print("params = ", params)
    print("lnprob = ", lnprob(params, *args))

    if str(sys.argv[1]) == "run":

        # burn in
        params = burnin(params, mods, args, t, niter=50000, nsteps=1,
                        clobber=False)
        print("initial_params =", params)

        # Run chains
        fn1 = str(sys.argv[2])
        fn = "{}_iso".format(fn1)
        run_multiple_chains(fn, params, mods, args, t, niter=5000, nsteps=1,
                            plot=False)

    if str(sys.argv[1]) == "combine":
        fn_list = ["0_iso", "1_iso", "2_iso", "3_iso"]
        fn = "combined_samples_iso_only"
        combine_samples(fn_list, fn, mods, params, plot=False)
