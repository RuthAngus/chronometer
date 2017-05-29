"""
Making a plot of gyrochronology ages vs isochronal ages.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import make_param_dict
from isochrones import StarModel
from isochrones.mist import MIST_Isochrone
import teff_bv as tbv
from isochrones.mist import MIST_Isochrone
import corner
import emcee
import time

plotpar = {'axes.labelsize': 18,
           'font.size': 10,
           'legend.fontsize': 18,
           'xtick.labelsize': 18,
           'ytick.labelsize': 18,
           'text.usetex': True}
plt.rcParams.update(plotpar)


def lnprob(par, mod):
    if 0 < par[0] < 100 and 0 < par[1] < 11 and -5 < par[2] < 5 and \
            0 < par[3] < 1e3 and 0 <= par[4] < 1:
        return mod.lnlike(par)
    else:
        return -np.inf


def calculate_isochronal_age(df, i, RESULTS_DIR):
    """
    Do the MCMC using isochrones.py
    """
    mist = MIST_Isochrone()
    param_dict = make_param_dict(df, i)
    param_dict = {k: param_dict[k] for k in param_dict if
                    np.isfinite(param_dict[k]).all()}

    p = np.zeros(5)
    p[0] = df.mass.values[i]
    p[1] = np.log10(1e9*df.age.values[i])
    p[2] = df.feh.values[i]
    p[3] = 1./(df.tgas_parallax.values[i])*1e3
    p[4] = df.Av.values[i]

    mod = StarModel(mist, **param_dict)

    nwalkers, nsteps, ndim, mult = 32, 500, len(p), 5
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

    # med, _15, _85 = (10**mod.samples.age_0_0.quantile([.5, .15, .85]))*1e-9

    # fig = mod.corner_physical()
    # print(np.shape(mod.samples))
    # print("Making corner plot")
    # fig = corner.corner(mod.samples[-1000:, :)

    # flat = np.vstack((mod.samples.mass_0_0, mod.samples.age_0_0)).T
                      # mod.samples.feh,
                      # mod.samples.distance,
                      # mod.samples.AV_0_0))
    fig = corner.corner(flat, labels=["Mass", "Age", "feh", "distance", "Av"])
    fig.savefig(os.path.join(RESULTS_DIR, "{}_corner.png".format(i)))

    plt.clf()
    plt.hist((10**mod.samples.age_0_0)*1e-9)
    plt.xlabel("$\mathrm{Age}$")
    plt.axvline(med)
    plt.axvline(_15)
    plt.axvline(_85)
    plt.savefig(os.path.join(RESULTS_DIR, "{}_hist.png".format(i)))

    return med, med - _15, _85 - med


def calculate_gyrochronal_ages(par, period, bv):
    """
    The gyro model.
    """
    a, b, c, n = par
    return (period / (a*(bv - c)**b))**(1./n) * 1e-3


def loop_over_stars(df, par, number, RESULTS_DIR):
    """
    Calculate gyro and iso ages for each star.
    Return lists of ages and uncertainties.
    """
    periods, teffs = df.prot.values[:number], df.teff.values[:number]
    fehs, loggs = df.feh.values[:number], df.logg.values[:number]
    bvs = tbv.teff2bv(teffs, loggs, fehs)
    gyro_age = calculate_gyrochronal_ages(par, periods, bvs)
    iso_ages, iso_errm, iso_errp, gyro_ages = [], [], [], []
    for i, star in enumerate(df.jmag.values[:number]):
        if df.prot.values[i] > 0.:
            print("Calculating iso age for ", i, "of",
                len(df.jmag.values[:number]), "...")

            # Check whether an age exists already
            fn = os.path.join(RESULTS_DIR, "{}.h5".format(i))

            # if os.path.exists(fn):
            #     d = pd.read_csv(fn)
            #     age, age_errm, age_errp = d.age.values, d.age_errm.values, \
            #         d.age_errp.values
            # else:
            age, age_errm, age_errp = \
                calculate_isochronal_age(df, i, RESULTS_DIR)
            d = pd.DataFrame({"age": [age], "age_errm": [age_errm],
                                "age_errp": [age_errp]})
            d.to_csv(fn)

            iso_ages.append(age)
            iso_errm.append(age_errm)
            iso_errp.append(age_errp)
            gyro_ages.append(gyro_age[i])
    return iso_ages, iso_errm, iso_errp, gyro_ages


def plot_gyro_age_against_iso_age(iso_ages, iso_errm, iso_errp, gyro_ages):
    plt.clf()
    xs = np.linspace(0, max(iso_ages), 100)
    plt.errorbar(iso_ages, gyro_ages, xerr=([iso_errm, iso_errp]), fmt="k.")
    plt.plot(xs, xs, ls="--")
    plt.xlabel("$\mathrm{Isochronal~age~(Gyr)}$")
    plt.ylabel("$\mathrm{Gyrochronal~age~(Gyr)}$")
    plt.subplots_adjust(bottom=0.15)
    plt.savefig("iso_vs_gyro.pdf")


def save_results():
    return


if __name__ == "__main__":

    # Preamble.
    DATA_DIR = "/Users/ruthangus/projects/chronometer/chronometer/data"
    RESULTS_DIR = "/Users/ruthangus/projects/chronometer/chronometer/iso_ages"
    # df = pd.read_csv(os.path.join(DATA_DIR, "kplr_tgas_periods.csv"))
    df = pd.read_csv(os.path.join(DATA_DIR, "action_data.csv"))

    par = np.array([.7725, .60, .4, .5189])
    iso_ages, iso_errm, iso_errp, gyro_ages = loop_over_stars(df, par, 1,
                                                              RESULTS_DIR)
    plot_gyro_age_against_iso_age(iso_ages, iso_errm, iso_errp, gyro_ages)
