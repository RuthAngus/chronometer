import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import scipy.misc as spm
import corner
import emcee


def model(par, x):
    return par[0] + par[1]*x


def lnlike(par, x, y, yerr):
    y_mod = model(par, x)
    return sum(-.5*((y_mod - y)/yerr)**2)


def metropolis_hastings(r, N, s, t, x, y, yerr):
    """
    params:
    -------
    r: (tuple or list or array)
        The parameters/position in probability space.
    N: (int)
        Number of samples.
    s: (int)
        Thinning index

    """
    print("inits = ", r)
    p = np.exp(lnlike(r, x, y, yerr))  # Initial probability.
    samples = []
    for i in np.arange(N):
        rn = r + np.random.normal(size=2)*t  # Take a step in parameter space.
        pn = np.exp(lnlike(r, x, y, yerr))  # calculate the probability.
        if pn >= p:  # If the new probability is greater than the previous p
            p = pn  # The probability is now the new one.
            r = rn  # adopt the new parameters.
        else:
            u = np.random.rand()  # Otherwise, random number between 0 and 1.
            if u < pn/p:  # if less than the ratio of new p to old p:
                p = pn  # prob is now new prob.
                r = rn  # adopt the new parameters.
        samples.append(r)
    return np.array(samples)

def MH(par, N, t, x, y, yerr):
    """
    params:
    -------
    par: (list)
        The parameters.
    args: *args
        Arguments to parse to the lnlike.
    N: (int)
        Number of samples.
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

    # Test on a straight line model
    x = np.arange(0, 10, .1)
    err = 2.
    yerr = np.ones_like(x) * err
    y = .7 + 2.5*x + np.random.randn(len(x))*err

    # Plot the data.
    plt.clf()
    plt.errorbar(x, y, yerr=yerr, fmt="k.")
    plt.savefig("data")

    # First, try brute force random sampling.
    m = np.random.randn(1000) + 2.5
    c = np.random.randn(1000) + .5
    args = np.array([x, y, yerr])
    Lm = np.array([np.exp(lnlike([.7, m[i]], x, y, yerr)) for i in
                   range(len(m))])
    Lc = np.array([np.exp(lnlike([c[i], 2.5], x, y, yerr)) for i in
                   range(len(m))])

    # Print and plot the results
    lc, lm = Lc == max(Lc), Lm == max(Lm)
    print(m[lm], c[lc])
    plt.clf()
    plt.plot(m, Lm, "k.")
    plt.axvline(2.5)
    plt.xlabel("m")
    plt.ylabel("likelihood")
    plt.savefig("corner_m")
    plt.clf()
    plt.plot(c, Lc, "k.")
    plt.axvline(.7)
    plt.xlabel("c")
    plt.ylabel("likelihood")
    plt.savefig("corner_c")

    print("Running emcee")
    ndim, nwalkers, nsteps = 2, 24, 10000
    p0 = [np.random.rand(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnlike, args=[x, y, yerr])
    sampler.run_mcmc(p0, nsteps)
    flat = np.reshape(sampler.chain, (nwalkers*nsteps, ndim))
    plt.clf()
    plt.hist(flat[:, 0], 100)
    plt.axvline(.7, color="HotPink")
    plt.savefig("emcee_corner")

    print("Running Metropolis Hastings 1")
    N = 1000000  # N samples
    s = 1  #
    pars = [.5, 2.5]  # initialisation
    t = .01
    samples1 = metropolis_hastings(pars, N, s, t, x, y, yerr)
    print(np.shape(samples1), "metropolis hastings")

    print("Running Metropolis Hastings 2")
    samples2 = MH(pars, N, t, x, y, yerr)
    print(np.shape(samples2), "mh")

    plt.clf()
    print(np.shape(flat))
    plt.hist(samples1[:, 0], 10, normed=True, alpha=.5, label="mh1")
    plt.hist(samples2[:, 0], 10, normed=True, alpha=.5, label="mh2")
    plt.hist(flat[:, 0], 10, normed=True, alpha=.5, label="emcee")
    plt.legend()
    plt.axvline(.7, color=".7", ls="--")
    plt.xlim(-1, 2)
    plt.savefig("corner")

    fig = corner.corner(flat)
    fig.savefig("triangle_emcee")

    fig = corner.corner(samples2)
    fig.savefig("triangle_mh")

    fig = corner.corner(samples1)
    fig.savefig("triangle_mh1")
