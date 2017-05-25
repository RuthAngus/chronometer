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


def lnlike(params, *args):
    """
    Probability of age and model parameters given rotation period and colour.
    Parameters to pass to the models are selected using the par_inds list in
    *args.
    Whether a 'gyro parameter only' or 'single star isochrone parameters only'
    gibbs step is determined according to the elements of par_inds.
    ----------
    params: (array)
        The array of log parameters: a, b, n, masses, ages, etc.
    args = mods, periods, period_errs, bvs, bv_errs, par_inds
    mods: (list)
        A list of starmodel objects, one for each star.
    par_inds: (array) args[-1]
        The indices of the parameters to vary.
    """
    mods, period, period_errs, bv, bv_errs, par_inds = args
    N, ngyro, nglob, nind = get_n_things(mods, params)
    gyro_lnlike, iso_lnlike = 0, 0
    if par_inds[0] == 0 and par_inds[1] == 1 and par_inds[2] == 2: # if gyro.
        # r = np.isfinite(period)
        # r_period, r_period_errs, r_bv = period[r], period_errs[r], bv[r]
        # gyro_lnlike = sum(-.5*((r_period - gyro_model(params, r_bv))
                                # /r_period_errs)**2)
        m = (bv > .4) * (np.isfinite(period))
        par_inds_mask = np.concatenate((par_inds[:ngyro],
                                        g_par_inds[ngyro:][m]))
        gyro_lnlike = sum(-.5*((period[m] - gyro_model(params[par_inds_mask],
                                                       bv[m]))
                                /period_errs[m])**2)

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


def lnprior(params, *args):
    """
    lnprior on all parameters.
    """
    par_inds = args[-1]

    # if gyro.
    if par_inds[0] == 0 and par_inds[1] == 1 and par_inds[2] == 2:
        g_prior = priors.lng_prior(params[:3])
        age_prior = sum([np.log(priors.age_prior(np.log10(1e9*np.exp(i))))
                        for i in params[3:]])
        feh_prior = 0.
        distance_prior = 0.
        mAv = True

    # if kinematics
    if par_inds[0] == 3:
        age_prior = sum([np.log(priors.age_prior(np.log10(1e9*np.exp(i))))
                        for i in params[1:]])
        if 0 < params[par_inds[0]] < 1e20:
            return age_prior
        else:
            return -np.inf

    # If individual stars
    else:
        g_prior = 0.
        age_prior = np.log(priors.age_prior(np.log10(1e9*np.exp(params[1]))))
        feh_prior = np.log(priors.feh_prior(params[2]))
        distance_prior = np.log(priors.distance_prior(np.exp(params[3])))
        mAv = (0 <= params[4]) * (params[4] < 1)  # Prior on A_v

    m = (-20 < params) * (params < 20)  # Broad bounds on all params.

    if sum(m) == len(m) and mAv:
        return g_prior + age_prior + feh_prior + distance_prior
    else:
        return -np.inf


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


def lnprob(params, *args):
    """
    The joint log-probability of age given gyro and iso parameters.
    params: (array)
        The parameter array.
    args: (list)
        args to pass to lnlike, including mods, a list of pre-computed star
        model objects.
    """
    return lnlike(params, *args) + lnprior(params, *args)


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
    if nsteps > 0:
        print("Acceptance fraction = ", accept/float(nsteps))
    return samples, par, probs


def MH_step(par, lnprob, t, *args, emc=False):
    """
    A single Metropolis step.
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
    This function tells the metropolis hastings what parameters to sample in
    and assembles the samples into an array. Because the ages are sampled
    twice, the gyro age parameters are tacked onto the end of this array.
    I'm not actually sure if this is the correct thing to do...
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
    nstars = len(args[0])
    n_parameter_sets = len(par_inds_list)

    # Final sample array
    all_samples = np.zeros((nsteps * niter, ndim + nstars))

    # Iterate over niter cycles of parameter sets.
    probs = []
    for i in range(niter):  # Loop over Gibbs repeats.
        print("Gibbs iteration ", i, "of ", niter)
        print("Current parameter values = ", par)
        for k in range(len(par_inds_list)):  # loop over parameter sets.
            args[-1] = par_inds_list[k]
            samples, par, pb = MH(par, lnprob, nsteps, t[k],
                                                   *args)
            if par_inds_list[k][0] == 0:  # save age samples separately
                all_samples[nsteps*i:nsteps*(i+1), par_inds_list[k][:3]] = \
                    samples[:, :3]
                all_samples[nsteps*i:nsteps*(i+1), -nstars:] = \
                    samples[:, 3:]
            else:
                all_samples[nsteps*i:nsteps*(i+1), par_inds_list[k]] = samples
            probs.append(pb)


    lnprobs = np.array([i for j in probs for i in j])
    return all_samples, lnprobs


# def estimate_covariance():
def estimate_covariance(nstars):
    """
    Return the covariance matrix of the emcee samples.
    If there are more stars than the three that were used to construct this
    matrix, repeat the last five columns and rows nstar times.
    """
    with h5py.File("emcee_posterior_samples.h5", "r") as f:
        samples = f["samples"][...]
    cov = np.cov(samples, rowvar=False)
    # return cov
    print(np.shape(cov))
    n = (np.shape(cov)[0] - 3)/5
    print(n)
    nadd = nstars - n
    print(nadd)
    star_cov_column = cov[:, -5:]
    print(np.shape(star_cov_column))
    for i in range(nadd):
        newcov = np.vstack((cov, star_cov_column))
        print(np.shape(newcov))
        newcov = np.hstack((cov, star_cov_column.T))
        print(np.shape(newcov))
    print(np.shape(newcov))
    assert 0
    return newcov


def find_optimum():
    """
    Return the median of the emcee samples.
    """
    with h5py.File("emcee_posterior_samples.h5", "r") as f:
        samples = f["samples"][...]
    ndim = np.shape(samples)[1]
    return np.array([np.median(samples[:, i]) for i in range(ndim)])


if __name__ == "__main__":

    # Use Metropolis hastings or emcee?
    run_MH = False
    # emcee_RESULTS_DIR = "/Users/ruthangus/projects/chronometer/chronometer/MH"
    RESULTS_DIR = "/Users/ruthangus/projects/chronometer/chronometer/emc"

    # Load the data for the initial parameter array.
    DATA_DIR = "/Users/ruthangus/projects/chronometer/chronometer/data"
    # d = pd.read_csv(os.path.join(DATA_DIR, "data_file.csv")
    d = pd.read_csv(os.path.join(DATA_DIR, "action_data.csv"))

    # Generate the initial parameter array and the mods objects from the data
    # The initial parameters
    global_params = np.array([.7725, .601, .5189, np.log(350.)])  # a, b, n, beta

    params, mods = pars_and_mods(d, global_params)
    # params = find_optimum()

    start = time.time()  # timeit

    # Set nsteps and niter.
    nsteps = 1000
    niter = 10
    N, ngyro, nglob, nind = get_n_things(mods, params)
    print(N, "stars")

    # Construct parameter indices for the different parameter sets.
    par_inds = np.arange(len(params))  # All
    age_par_inds = par_inds[nglob+N:nglob+2*N]
    gyro_par_inds = np.concatenate((par_inds[:ngyro], age_par_inds))
    kin_par_inds = list(age_par_inds)
    kin_par_inds.insert(0, par_inds[ngyro])
    par_inds_list = [gyro_par_inds, np.array(kin_par_inds)]
    for i in range(N):
        par_inds_list.append(par_inds[nglob+i::N])  # Iso stars.

    # Create the covariance matrices.
    # t = estimate_covariance(N)
    # ts = []
    # for i, par_ind in enumerate(par_inds_list):
    #     ti = np.zeros((len(par_ind), len(par_ind)))
    #     for j, ind in enumerate(par_ind):
    #         ti[j] = t[ind][par_ind]
    #     ts.append(ti)

    # Sample posteriors using either MH gibbs or emcee
    if run_MH:
        args = [mods, d.prot.values, d.prot_err.values, d.bv.values,
                d.bv_err.values, par_inds_list]
        flat, lnprobs = gibbs_control(params, lnprob, nsteps, niter, ts,
                                    par_inds_list, args)

        # Throw away _number_ Gibbs iterations as burn in. FIXME
        number = 2
        burnin = nsteps * number
        flat = flat[burnin:, :]

    else:
        emcee_args = [mods, d.prot.values,
                      d.prot_err.values, d.bv.values,
                      d.bv_err.values, d.Jz, d.Jz_err, par_inds]
        nwalkers, nsteps, ndim = 64, 10000, len(params)
        p0 = [1e-4*np.random.rand(ndim) + params for i in range(nwalkers)]
        sampler = emcee.EnsembleSampler(nwalkers, ndim, emcee_lnprob,
                                        args=emcee_args)
        print("burning in...")
        pos, _, _ = sampler.run_mcmc(p0, 5000)
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
        assert 0

    end = time.time()
    print("Time taken = ", (end - start)/60, "minutes")

    print("Plotting results and traces")
    plt.clf()
    if run_MH:
        for j in range(int(len(lnprobs)/nsteps - 1)):
            x = np.arange(j*nsteps, (j+1)*nsteps)
            plt.plot(x, lnprobs[j*nsteps: (j+1)*nsteps])
    else:
        plt.plot(sampler.lnprobability.T)
    plt.xlabel("Time")
    plt.ylabel("ln (probability)")
    plt.savefig(os.path.join(RESULTS_DIR, "prob_trace"))

    labels = ["$a$", "$b$", "$n$", "$\ln(Mass_1)$", "$\ln(Mass_2)$",
              "$\ln(Mass_3)$", "$\ln(Age_{1,i})$", "$\ln(Age_{2,i})$",
              "$\ln(Age_{3,i})$", "$[Fe/H]_1$", "$[Fe/H]_2$", "$[Fe/H]_3$",
              "$\ln(D_1)$", "$\ln(D_2)$", "$\ln(D_3)$", "$A_{v1}$",
              "$A_{v2}$", "$A_{v3}$", "$\ln(Age_{1,g})$",
              "$\ln(Age_{2,g})$", "$\ln(Age_{3,g})$"]

    ages = np.zeros((np.shape(flat)[0]*2, N))
    for i in range(N):
        ages[:, i] = np.concatenate((flat[:, 3+N+i], flat[:, 3+5*N+i]))
        plt.clf()
        plt.plot(ages[:, i])
        plt.ylabel("age {}".format(i))
        plt.savefig(os.path.join(RESULTS_DIR, "age_{}_chain".format(i)))

    # Plot chains
    ndim = len(params)
    for i in range(ndim):
        plt.clf()
        if run_MH:
            for j in range(niter - number):
                x = np.arange(j*nsteps, (j+1)*nsteps)
                plt.plot(x, flat[j*nsteps: (j+1)*nsteps, i].T)
                plt.ylabel(labels[i])
        else:
            plt.plot(sampler.chain[:, :,  i].T, alpha=.5)
        plt.savefig(os.path.join(RESULTS_DIR, "{}_trace".format(i)))

    print("Making corner plot")
    truths = [.7725, .601, .5189, np.log(1), None, None, np.log(4.56),
              np.log(2.5), np.log(2.5), 0., None, None, np.log(10),
              np.log(2400), np.log(2400), 0., None, None, np.log(4.56),
              np.log(2.5), np.log(2.5)]
    fig = corner.corner(flat, truths=truths, labels=labels)
    fig.savefig(os.path.join(RESULTS_DIR, "demo_corner_gibbs"))

    f = h5py.File(os.path.join(RESULTS_DIR, "samples.h5"), "w")
    data = f.create_dataset("samples", np.shape(flat))
    data[:, :] = flat
    f.close()
