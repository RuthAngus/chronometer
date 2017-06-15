"""
Making a plot of gyrochronology ages vs isochronal ages.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import make_param_dict
from isochrones import StarModel
import teff_bv as tbv
from isochrones.mist import MIST_Isochrone
import corner
import emcee
import time
import h5py
import priors

plotpar = {'axes.labelsize': 18,
           'font.size': 10,
           'legend.fontsize': 18,
           'xtick.labelsize': 18,
           'ytick.labelsize': 18,
           'text.usetex': True}
plt.rcParams.update(plotpar)


def lnprior(par):
    age_prior = np.log(priors.age_prior(par[1]))
    feh_prior = np.log(priors.feh_prior(par[2]))
    distance_prior = np.log(priors.distance_prior(np.exp(par[3])))
    return age_prior + feh_prior + distance_prior


def lnprob(par, mod):
    if 0 < par[0] < 100 and 0 < par[1] < 11 and -5 < par[2] < 5 and \
            0 < par[3] < 1e10 and 0 <= par[4] < 1:
        prob = mod.lnlike(par) + lnprior(par)
        if np.isfinite(prob):
            return prob
        else:
            return -np.inf
    else:
        return -np.inf


def calculate_isochronal_age(param_dict, i, RESULTS_DIR):
    """
    Do the MCMC using isochrones.py
    """
    mist = MIST_Isochrone()

    # Initial values
    p = np.zeros(5)
    p[0] = 1.  # df.mass.values[i]
    p[1] = 9.  # np.log10(1e9*df.age.values[i])
    p[2] = 0.  # df.feh.values[i]
    p[3] = 100. # 1./(df.tgas_parallax.values[i])*1e3
    p[4] = 0.  # df.Av.values[i]

    if type(param_dict) == dict:
        mod = StarModel(mist, **param_dict)
    else:
        mod = param_dict

    # Run emcee
    nwalkers, nsteps, ndim, mult = 32, 10000, len(p), 5
    p0 = [1e-4*np.random.rand(ndim) + p for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[mod])
    print("burning in...")
    start = time.time()
    pos, _, _ = sampler.run_mcmc(p0, nsteps)
    end = time.time()
    print("Time = ", (end - start)/60, "minutes")
    print("Predicted run time = ", (end - start)/60 * mult, "minutes")
    sampler.reset()
    print("production run...")
    start = time.time()
    sampler.run_mcmc(pos, mult*nsteps)
    end = time.time()
    print("Time = ", (end - start)/60, "minutes")
    flat = np.reshape(sampler.chain, (nwalkers*nsteps*mult, ndim))

    # Plot figure
    flat[:, 1] = 10**(flat[:, 1])*1e-9
    fig = corner.corner(flat, labels=["Mass", "Age", "feh", "distance", "Av"])
    fig.savefig(os.path.join(RESULTS_DIR, "{}_corner.png".format(i)))

    f = h5py.File(os.path.join(RESULTS_DIR, "{}.h5".format(i)), "w")
    data = f.create_dataset("samples", np.shape(flat))
    data[:, :] = flat
    f.close()

    med = np.percentile(flat[:, 1], 50)
    lower = np.percentile(flat[:, 1], 16)
    upper = np.percentile(flat[:, 1], 84)
    logerrm, logerrp = med - lower, upper - med
    errp = logerrp/med
    errm = logerrm/med
    return med, errm, errp, flat[:, 1]
    # errp = 10**(logerrp/med)*1e-9
    # errm = 10**(logerrm/med)*1e-9
    # return (10**med)*1e-9, errm, errp, flat[:, 1]


def calculate_gyrochronal_ages(par, period, bv):

    """
    The gyro model.
    """
    a, b, c, n = par
    return (period / (a*(bv - c)**b))**(1./n) * 1e-3


def loop_over_stars(df, par, number, RESULTS_DIR, clobber=False):
    """
    Calculate gyro and iso ages for each star.
    Return lists of ages and uncertainties.
    """
    try:
        bvs = df.bv.values
    except:
        teffs = df.teff.values[:number]
        fehs, loggs = df.feh.values[:number], df.logg.values[:number]
        bvs = tbv.teff2bv(teffs, loggs, fehs)

    periods = df.prot.values[:number]
    gyro_age = calculate_gyrochronal_ages(par, periods, bvs[:number])

    iso_ages, iso_errm, iso_errp, gyro_ages = [], [], [], []
    for i, star in enumerate(df.jmag.values[:number]):
        if df.prot.values[i] > 0. and bvs[i] > .4:
            print("Calculating iso age for ", i, "of",
                len(df.jmag.values[:number]), "...")

            # Check whether an age exists already
            fn = os.path.join(RESULTS_DIR, "{}.h5".format(i))

            if clobber:
                param_dict = make_param_dict(df, i)
                param_dict = {k: param_dict[k] for k in param_dict if
                                np.isfinite(param_dict[k]).all()}

                age, age_errm, age_errp, samps = \
                    calculate_isochronal_age(param_dict, i, RESULTS_DIR)
                d = pd.DataFrame({"age": [age], "age_errm": [age_errm],
                                    "age_errp": [age_errp]})
                d.to_csv(fn)
            else:
                if os.path.exists(fn):
                    d = pd.read_csv(fn)
                    age, age_errm, age_errp = d.age.values, \
                        d.age_errm.values, d.age_errp.values
                else:
                    param_dict = make_param_dict(df, i)
                    param_dict = {k: param_dict[k] for k in param_dict if
                                    np.isfinite(param_dict[k]).all()}

                    age, age_errm, age_errp, samps = \
                        calculate_isochronal_age(param_dict, i, RESULTS_DIR)
                    d = pd.DataFrame({"age": [age], "age_errm": [age_errm],
                                        "age_errp": [age_errp]})
                    d.to_csv(fn)

            iso_ages.append(age)
            iso_errm.append(age_errm)
            iso_errp.append(age_errp)
            gyro_ages.append(gyro_age[i])
    return iso_ages, iso_errm, iso_errp, gyro_ages


def plot_gyro_age_against_iso_age(iso_ages, iso_errm, iso_errp, gyro_ages,
                                  fn):
    # ages = np.array([3.5, 6.5, 1., 10, 4.5])
    xs = np.linspace(0, max(iso_ages), 100)

    plt.clf()
    plt.plot(xs, xs, ls="--")
    plt.errorbar(gyro_ages, iso_ages, yerr=([iso_errm, iso_errp]), fmt="k.")
    plt.xlabel("$\mathrm{Gyrochronal~age~(Gyr)}$")
    plt.ylabel("$\mathrm{Isochronal~age~(Gyr)}$")
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(os.path.join(RESULTS_DIR, fn))


if __name__ == "__main__":

    # Preamble.
    DATA_DIR = "/Users/ruthangus/projects/chronometer/chronometer/data"
    # RESULTS_DIR = "/Users/ruthangus/projects/chronometer/chronometer/iso_ages"
    RESULTS_DIR = "/Users/ruthangus/projects/chronometer/chronometer/"\
        "fake_iso_ages"
    # df = pd.read_csv(os.path.join(DATA_DIR, "kplr_tgas_periods.csv"))
    # df = pd.read_csv(os.path.join(DATA_DIR, "action_data.csv"))
    df = pd.read_csv(os.path.join(DATA_DIR, "fake_data.csv"))

    par = np.array([.7725, .60, .4, .5189])
    iso_ages, iso_errm, iso_errp, gyro_ages = loop_over_stars(df, par, 3,
                                                              RESULTS_DIR,
                                                              clobber=True)
    plot_gyro_age_against_iso_age(iso_ages, iso_errm, iso_errp, gyro_ages,
                                  "fake_data")
                                  # "real_data_more")
