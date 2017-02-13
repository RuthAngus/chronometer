
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.misc as spm
from isochrones import StarModel
from isochrones.mist import MIST_Isochrone

from simple_gibbs import gibbs
import emcee


def gc_model(params, data):
    """
    Given an age and a B-V colour, predict a rotation period.
    Returns log(age) in Myr.
    """
    a, b, n = np.exp(params)
    p, bv = data
    return (np.log(p) - np.log(a) - b*np.log(bv - .4))/n


def gc_lnlike(params, data, age, age_err):
    """
    Probability of rotation period and colour given age
    """
    model = gc_model(params, data)
    return sum(-.5*((age - model)/age_err)**2)


def gc_lnprior(params):
    """
    Some simple, uninformative prior over the parameters.
    """
    if (-10 < params[0] < 10) and (-10 < params[1] < 10) and \
            (-10 < params[2] < 10):
        return 0.
    else:
        return -np.inf


def gc_lnprob(params, data, age, age_err):
    return gc_lnlike(params, data, age, age_err) + gc_lnprior(params)


def iso_lnlike(data, age, age_err):
    """
    Some isochronal likelihood function.
    """
    logg, teff, feh = data
    return 0.


def iso_lnprior(params):
    return 0.


def lnprob(data, params, age, age_err):
    logg, teff, feh, t, bv = data
    return iso_lnlike([logg, teff, feh], age, age_err) + \
        gc_lnlike([t, bv], params, age, age_err) + gc_lnprior(params) + \
        iso_lnprior(params)


if __name__ == "__main__":

    DATA_DIR = "/Users/ruthangus/projects/chronometer/chronometer/data"

    params = np.log([.7725, .601, .5189])

    # load data.
    cs = pd.read_csv(os.path.join(DATA_DIR, "clusters.csv"))
    m = (.6 < cs.bv.values) * (cs.bv.values < .7)
    cs = cs.iloc[m]
    periods = cs.period.values
    bvs = cs.bv.values
    ages = cs.age.values
    age_err = cs.age_err.values

    print(gc_lnprob(params, [periods, bvs], ages, age_err))

    # Run emcee
    ndim, nwalkers, nsteps = len(params), 24, 10000
    p0 = [1e-4*np.random.rand(ndim) + params for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, gc_lnprob,
                                    args=[[periods, bvs], ages, age_err])

    pos, _, _ = sampler.run_mcmc(p0, 500)
    sampler.reset()
    sampler.run_mcmc(pos, nsteps)
    flat = np.reshape(sampler.chain, (nwalkers*nsteps, ndim))
    fig = corner.corner(flat)
    fig.savefig("corner_test")
    results = [np.median(flat[:, 0]), np.median(flat[:, 1]),
               np.median(flat[:, 2])]
    print(results)

    # Plot data with model
    plt.clf()
    loga = gc_model(params, [periods, bvs])
    plt.plot(loga, np.log(periods))
    loga = gc_model(results, [periods, bvs])
    plt.plot(loga, np.log(periods))
    plt.plot(np.log(ages*1e3), np.log(periods), "k.")
    plt.xlabel("age")
    plt.ylabel("period")
    plt.savefig("model")

    # mist = MIST_Isochrone()
    # mod = StarModel(mist, Teff=(5700, 100), logg=(4.5, 0.1), feh=(0.0, 0.1))
    # # pardict = {"Teff": 5700, "Teff_err": 100, "logg": 4.5, "logg_err": .1,
    # #            "feh": 0., "feh_err": .1}
    # p0 = [0.8, 9.5, 0.0, 200, 0.2]
    # print(mod.lnlike(p0))
    # assert 0
