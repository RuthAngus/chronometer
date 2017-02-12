import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner


def model(pars, fixed, x, fit):
    """
    The Model. A straight line plus a Gaussian.
    params:
    ------
    pars: (tuple, list or string)
        The parameters to vary.
    fixed: (tuple, list or string)
        The fixed parameters.
    x: (array)
        The x-array
    fit: (int)
        Either 0 or 1. If 0, the line parameters are read as pars and the
        Gaussian parameters are read as fixed. The reverse is true if 1.
    returns: (array)
        The computed y-values of the model.
    """
    if fit == 0:
        a, b = np.exp(pars)
        mu, sig = np.exp(fixed)
    elif fit == 1:
        mu, sig = np.exp(pars)
        a, b = np.exp(fixed)
    return a + b*x + np.exp(-(x - mu)**2/(2.*sig**2))


def line_lnprob(pars0, pars1, x, y, yerr):
    """
    The log-prob of the line model.
    pars0: (tuple, list or string)
        The line parameters, pars0 = [a, b]
    pars1: (tuple, list or string)
        The Gaussian parameters, pars1 = [mu, sigma]
    x: (array)
        The x-array
    y: (array)
        The observed y-values
    yerr: (array)
        The observed y-uncertainties.
    returns: (float)
        The log-probability.
    """
    mod = model(pars0, pars1, x, 0)
    return sum(-((y - mod)**2/(2.*yerr**2))) + line_lnprior(pars0)


def line_lnprior(pars0):
    """
    The log-prior for the parameters of the line model.
    """
    a, b = pars0
    if -10 < a < 10 and -10 < b < 10:
        return 0.
    else:
        return -np.inf


def Gauss_lnprob(pars1, pars0, x, y, yerr):
    """
    The log-prob of the Gaussian model.
    """
    mod = model(pars1, pars0, x, 1)
    return sum(-((y - mod)**2/(2.*yerr**2))) + Gauss_lnprior(pars1)


def Gauss_lnprior(pars1):
    """
    The log-prior for the parameters of the Gaussian model.
    """
    mu, sig = pars1
    if -10 < mu < 10 and -10 < sig < 10:
        return 0.
    else:
        return -np.inf


def gibbs_internal(pars0, pars1, x, y, yerr, fit):
    """
    A simple Gibbs-sampling skeleton. Does a single MCMC iteration.
    params:
    ------
    pars0: (tuple, list or string)
        The line parameters, pars0 = [a, b]
    pars1: (tuple, list or string)
        The Gaussian parameters, pars1 = [mu, sigma]
    x: (array)
        The x-array
    y: (array)
        The observed y-values
    yerr: (array)
        The observed y-uncertainties.
    fit: (int)
        Either 0 or 1. If 0, the line parameters are varied and the
        Gaussian parameters are fixed. The reverse is true if 1.
    returns: (2-d array)
        The sample array of the conditional PDF that is being sampled.
    """
    if fit == 0:
        lnprob = line_lnprob
        par_init, fixed = pars0, pars1
    elif fit == 1:
        lnprob = Gauss_lnprob
        par_init, fixed = pars1, pars0
    print("varying = ", np.exp(par_init), "fixed = ", np.exp(fixed))

    ndim, nwalkers, nsteps = 2, 24, 10000
    p0 = [1e-4*np.random.rand(ndim) + par_init for i in range(nwalkers)]

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[fixed, x, y,
                                                                  yerr])
    pos, _, _ = sampler.run_mcmc(p0, 500)
    sampler.reset()
    sampler.run_mcmc(pos, nsteps)
    flat = np.reshape(sampler.chain, (nwalkers*nsteps, ndim))
    fig = corner.corner(flat)
    fig.savefig("corner{}".format(fit))

    par_result = [np.median(flat[:, 0]), np.median(flat[:, 1])]
    model_result = model(par_result, fixed, x, fit)

    plt.clf()
    plt.plot(x, y, "k.")
    plt.plot(x, model_result)
    plt.savefig("result{}".format(fit))
    return par_result, flat


def gibbs(par_init0, par_init1, x, y, yerr, niter):
    """
    The Gibbs wrapper. Given two conditionally independent models, sample the
    parameters of each model, one at a time and iterate back and forth for
    niter iterations.
    """

    p0, p1 = par_init0, par_init1
    for i in range(niter):
        print("Model {0}, Iteration {1}".format(0, i))
        p0, flat0 = gibbs_internal(p0, p1, x, y, yerr, 0)
        print("Model {0}, Iteration {1}".format(1, i))
        p1, flat1 = gibbs_internal(p0, p1, x, y, yerr, 1)

    print(np.shape(flat0), np.shape(flat1))
    flat = np.concatenate((flat0, flat1), axis=1)
    fig = corner.corner(flat)
    fig.savefig("corner")

    return p0, p1, flat0, flat1

if __name__ == "__main__":

    x = np.linspace(0, 10, 100)
    pars0 = np.log([2., .5])
    pars1 = np.log([5., .3])
    err = .1
    yerr = np.ones_like(x) * err
    y = model(pars0, pars1, x, 0) + np.random.randn(len(x)) * yerr

    plt.clf()
    plt.plot(x, y, "k.")
    plt.savefig("data")

    line_par_init = np.log([3.1, 1.1])
    Gauss_par_init = np.log([3.2, 1.2])

    line_results, Gauss_results, flat_line, flat_Gauss = gibbs(line_par_init,
                                                               Gauss_par_init,
                                                               x, y, yerr, 2)
