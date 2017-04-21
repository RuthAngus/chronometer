"""
Test the metropolis hastings algorithm.
"""

import numpy as np
import chronometer as gc
import matplotlib.pyplot as plt
import corner
import emcee


def model(par, x):
    return par[0] + par[1]*x


def lnlike(par, x, y, yerr, par_inds):
    y_mod = model(par, x)
    return sum(-.5*((y_mod - y)/yerr)**2)


def test_metropolis_hastings():
    # Straight line model
    x = np.arange(0, 10, .1)
    err = 2.
    yerr = np.ones_like(x) * err
    y = .7 + 2.5*x + np.random.randn(len(x))*err

    # Plot the data.
    plt.clf()
    plt.errorbar(x, y, yerr=yerr, fmt="k.")
    plt.savefig("data")

    print("Running Metropolis Hastings")
    N = 1000000  # N samples
    pars = np.array([.5, 2.5])  # initialisation
    t = np.array([.01, .01])
    par_inds = np.arange(len(pars))
    args = [x, y, yerr, par_inds]
    samples, par, probs = gc.MH(pars, lnlike, N, t, *args)

    results = [np.percentile(samples[:, i], 50) for i in range(2)]
    upper = [np.percentile(samples[:, i], 64) for i in range(2)]
    lower = [np.percentile(samples[:, i], 15) for i in range(2)]

    print(lower, "lower")
    print(results, "results")
    print(upper, "upper")
    assert lower < results
    assert results < upper

    plt.clf()
    plt.errorbar(x, y, yerr=yerr, fmt="k.")
    plt.plot(x, results[0] + results[1]*x)
    plt.savefig("test")

    fig = corner.corner(samples, truths=[.7, 2.5], labels=["m", "c"])
    fig.savefig("corner_MH_test")

    plt.clf()
    plt.plot(probs)
    plt.savefig("prob_test")

if __name__ == "__main__":
    test_metropolis_hastings()
