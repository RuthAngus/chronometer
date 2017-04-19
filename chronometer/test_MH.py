"""
Test the metropolis hastings algorithm.
"""

import numpy as np
import gibbs_chronometer as gc
import matplotlib.pyplot as plt
import corner
import emcee


def model(par, x):
    return par[0] + par[1]*x


def lnlike(par, x, y, yerr):
    y_mod = model(par, x)
    return sum(-.5*((y_mod - y)/yerr)**2)


def MH(par, N, t, x, y, yerr):
    """
    params:
    -------
    par: (list)
        The parameters.
    x, y, yerr: (arrays)
        The data
    N: (int)
        Number of samples.
    t: (float)
        The std of the proposal distribution.
    """
    samples = np.zeros((N, len(par)))
    for i in range(N):
        newp = par + np.random.randn(len(par))*t
        alpha = np.exp(lnlike(newp, x, y, yerr))/np.exp(lnlike(par, x, y,
                                                               yerr))
        if alpha > 1:
            par = newp*1
        else:
            u = np.random.uniform(0, 1)
            if alpha > u:
                par = newp*1
        samples[i, :] = par
    return samples


if __name__ == "__main__":

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
    pars = [.5, 2.5]  # initialisation
    t = .01
    samples = MH(pars, N, t, x, y, yerr)

    results = [np.percentile(samples[:, i], 50) for i in range(2)]

    plt.clf()
    plt.errorbar(x, y, yerr=yerr, fmt="k.")
    plt.plot(x, results[0]*y + results[1])
    plt.savefig("test")

    fig = corner.corner(samples, truths=[2.5, .7], labels=["m", "c"])
    fig.savefig("corner_MH_test")
