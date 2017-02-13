
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.misc as spm
from isochrones import StarModel
from isochrones.mist import MIST_Isochrone


def gc_model(data, params):
    """
    Given an age and a B-V colour, predict a rotation period.
    Returns log(age) in Myr.
    """
    a, b, c, n = np.exp(params)
    p, bv = data
    return (np.log(p) - np.log(a) - b*np.log(bv - c))/n
    # return (p/((bv - c)**b))**(1./n) * 1e-3


def gc_lnlike(data, age, params, age_err):
    """
    Probability of rotation period and colour given age
    """
    model = gc_model(data, params)
    return sum(-.5*((age - model)/age_err)**2)


def gc_lnprior(params):
    """
    Some simple, uninformative prior over the parameters.
    """
    return 0.


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
    mist = MIST_Isochrone()
    mod = StarModel(mist, Teff=(5700, 100), logg=(4.5, 0.1), feh=(0.0, 0.1))
    # pardict = {"Teff": 5700, "Teff_err": 100, "logg": 4.5, "logg_err": .1,
    #            "feh": 0., "feh_err": .1}
    p0 = [0.8, 9.5, 0.0, 200, 0.2]
    print(mod.lnlike(p0))

    assert 0
    DATA_DIR = "/Users/ruthangus/projects/chronometer/chronometer/data"

    periods = np.linspace(0, 100, 100)
    bvs = np.ones(100) * .65
    params = np.log([.7725, .601, .4, .5189])
    loga = gc_model([periods, bvs], params)

    # load data.
    cs = pd.read_csv(os.path.join(DATA_DIR, "clusters.csv"))
    m = (.6 < cs.bv.values) * (cs.bv.values < .7)
    cs = cs.iloc[m]

    plt.clf()
    plt.plot(loga, np.log(periods))
    plt.plot(np.log(cs.age*1e3), np.log(cs.period), "k.")
    plt.xlabel("age")
    plt.ylabel("period")
    plt.savefig("model")

    print(gc_lnlike([periods, bvs], 4.56, params, .1))
