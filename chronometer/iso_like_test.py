# Assembling likelihood functions for Gyrochronology, isochrones, dynamics
# and asteroseismology.

import matplotlib.pyplot as plt
import time
import numpy as np
from isochrones import StarModel
from isochrones.mist import MIST_Isochrone

import emcee
import corner


def iso_lnlike(lnparams, mod):
    """
    Some isochronal likelihood function.
    parameters:
    ----------
    params: (array)
        The array of parameters: log(mass), log(age in Gyr), metallicity,
        log(distance), extinction.
    mod: (object)
        An isochrones.py starmodel object.
    """
    lnmass, lnage, feh, lnd, Av = lnparams
    params = np.array([np.exp(lnmass), np.log10(1e9*np.exp(lnage)), feh,
                       np.exp(lnd), Av])
    return mod.lnlike(params)


def lnprior(params):
    """
    Some simple, uninformative prior over the parameters.
    age, mass, feh, distance, Av = params
    """
    if -10 < params[0] < 11 and -10 < params[1] < 10 and \
            -10 < params[2] < 10 and -10 < params[3] < 11 and \
            -10 < params[4] < 10:
        return 0.
    else:
        return -np.inf


def lnprob(params, mod):
    """
    The joint log-probability of age given gyro and iso parameters.
    mod: (list)
        list of pre-computed star model objects.
    """

    return iso_lnlike(params, mod) + lnprior(params)


def probtest(xs, i):
    lps = []
    p = params + 0
    for x in xs:
        p[i] = x
        lp = lnprob(p, mod)
        lps.append(lp)
    plt.clf()
    plt.plot(xs, lps)
    plt.xlabel("X")
    plt.ylabel("lnprob")
    plt.savefig("probs")


if __name__ == "__main__":

    age = 4.56
    mass, feh, d, Av = 1., 0., 10., 0.
    params = np.array([np.log(mass), np.log(age), feh, np.log(d), Av])

    # test on the Sun at 10 pc first.
    J, J_err = 3.711, .01  # absolute magnitudes/apparent at D = 10pc
    H, H_err = 3.453, .01
    K, K_err = 3.357, .01

    bands = dict(J=(J, J_err), H=(H, H_err), K=(K, K_err),)
    parallax = (.1, .001)

    # test the iso_lnlike
    mist = MIST_Isochrone()

    # iso_lnlike preamble.
    start = time.time()
    mod = StarModel(mist, J=(J, J_err), H=(H, H_err), K=(K, K_err),
                    parallax=(.1, .001))
    p0 = np.array([params[0], params[1], params[2], params[3], params[4]])


    # Timing the lhf.
    start = time.time()
    mod.lnlike(p0)
    end = time.time()
    print("preamble time = ", end - start)

    start = time.time()
    print("iso_lnlike = ", iso_lnlike(params, mod))
    end = time.time()
    print("lhf time = ", end - start)

    # test the lnprob.
    print("lnprob = ", lnprob(params, mod))

    # ages = np.log(np.arange(1., 10., 1))
    # masses = np.log(np.arange(.1, 2., .1))
    # fehs = np.arange(-.1, .1, .01)
    # ds = np.log(np.arange(8, 20, 1))
    # Avs = np.arange(.1, .5, .01)
    # probtest(ages, 1)
    # probtest(masses, 0)
    # probtest(fehs, 2)
    # probtest(ds, 3)
    # probtest(Avs, 4)
    # assert 0

    # Run MCMC
    nwalkers, nsteps, ndim = 64, 1000, len(params)
    p0 = [1e-4*np.random.rand(ndim) + params for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[mod])

    print("burning in...")
    pos, _, _ = sampler.run_mcmc(p0, 10)
    sampler.reset()

    print("production run...")
    sampler.run_mcmc(pos, nsteps)
    flat = np.reshape(sampler.chain, (nwalkers*nsteps, ndim))
    labels = ["$Age$", "$Mass$", "$[Fe/H]$", "$D$", "$A_v$"]
    fig = corner.corner(flat)
    fig.savefig("corner_iso")

    print(np.shape(flat))
    for i in range(ndim):
        plt.clf()
        plt.plot(flat[:, i], "k", alpha=.5)
        plt.ylabel(labels[i])
        plt.savefig("{}_trace".format(i))
