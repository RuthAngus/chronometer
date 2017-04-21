"""
Test the metropolis hastings algorithm.
"""

import numpy as np
import chronometer as gc
import matplotlib.pyplot as plt
import corner
import emcee


def line_model(par, x):
    return par[0] + par[1]*x


def gauss_model(par, x):
    return par[2]*np.exp(-.5*((par[0] - x)/par[1])**2)


def lnlike(par, x, y, yerr, par_inds):
    """
    The lnlike function determines which model to use according to which
    parameters are being parsed to it.
    """
    line_lnlike, gauss_lnlike = 0, 0
    if len(par_inds) == 2:
        y_mod = line_model(par, x)
        line_lnlike = sum(-.5*((y_mod - y)/yerr)**2)
    elif len(par_inds) == 3:
        y_mod = gauss_model(par, x)
        gauss_lnlike = sum(-.5*((y_mod - y)/yerr)**2)
    elif len(par_inds) == 5:
        y_mod = line_model(par, x)
        line_lnlike = sum(-.5*((y_mod - y)/yerr)**2)
        y_mod = gauss_model(par, x)
        gauss_lnlike = sum(-.5*((y_mod - y)/yerr)**2)
    return line_lnlike + gauss_lnlike + lnprior(par)


def lnprior(par):
    m = (0 < par) * (par < 15)
    if sum(m) == len(m):
        return 0.
    else:
        return -np.inf


def test_metropolis_hastings():

    par = np.array([.7, 2.5, 2, .5, 10])

    # Straight line model
    x = np.arange(0, 10, .1)
    err = 2.
    yerr = np.ones_like(x) * err
    y = line_model(par[:2], x) + np.random.randn(len(x))*err + \
        gauss_model(par[2:], x)

    # Plot the data.
    plt.clf()
    plt.errorbar(x, y, yerr=yerr, fmt="k.")
    plt.savefig("data")

    print("Running Metropolis Hastings")
    nsteps, niter = 100000, 6  # N samples
    t = np.array([.01, .01, .01, .01, .01])
    par_ind_list = [np.array([0, 1, 2, 3, 4]), np.array([0, 1]),
                    np.array([2, 3, 4])]
    # par_ind_list = [np.array([0, 1, 2, 3, 4])]

    args = np.array([x, y, yerr, par_ind_list[0]])
    samples, lnprobs = gc.gibbs_control(par, lnlike, nsteps, niter, t,
                                        par_ind_list, args)

    # Throw away _number_ Gibbs iterations as burn in.
    number = 3
    burnin = nsteps * number * 2
    samples = samples[burnin:, :]

    results = [np.percentile(samples[:, i], 50) for i in range(len(par))]
    upper = [np.percentile(samples[:, i], 64) for i in range(len(par))]
    lower = [np.percentile(samples[:, i], 15) for i in range(len(par))]

    print(lower, "lower")
    print(results, "results")
    print(upper, "upper")
    assert lower < results
    assert results < upper

    plt.clf()
    plt.errorbar(x, y, yerr=yerr, fmt="k.")
    plt.plot(x, line_model(results[:2], x) + gauss_model(results[2:], x))
    plt.savefig("test")

    fig = corner.corner(samples, truths=par, labels=["m", "c", "mu", "std",
                                                     "A"])
    fig.savefig("corner_MH_test")

    plt.clf()
    plt.plot(lnprobs)
    plt.savefig("prob_test")

if __name__ == "__main__":
    test_metropolis_hastings()
