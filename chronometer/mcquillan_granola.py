# Make a plot of age vs J_z for Kepler-TGAS.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py

import os


plotpar = {'axes.labelsize': 18,
           'font.size': 10,
           'legend.fontsize': 18,
           'xtick.labelsize': 18,
           'ytick.labelsize': 18,
           'text.usetex': True}
plt.rcParams.update(plotpar)


def calc_bin_dispersion(age, jz, nbins):
    hist_age, bins = np.histogram(age, nbins)  # make histogram
    dispersions, Ns = [], []
    m = age < bins[0]
    dispersions.append(RMS(jz[m]))
    Ns.append(len(age[m]))
    for i in range(len(bins)-1):
        m = (bins[0] < age) * (age < bins[i+1])
        if len(age[m]):
            dispersions.append(RMS(jz[m]))
            Ns.append(len(age[m]))
    return bins, np.array(dispersions), np.array(Ns)


def RMS(x):
    return (np.median(x**2))**.5


def dispersion(ages, Jzs, minage, maxage):
    """
    Dispersion in a single bin.
    """
    m = (minage < ages) * (ages < maxage)
    return RMS(Jzs[m]), len(ages[m])


def x_and_y(ages, Jzs):
    xs = np.linspace(min(ages), max(ages), 1000)
    ys = []
    for x in xs:
        y, N = dispersion(ages, Jzs, x-.5, x+.5)
        ys.append(y)
    return xs, ys


if __name__ == "__main__":
    DATA_DIR = "/Users/ruthangus/granola/granola/data"
    d = pd.read_csv("ages_and_actions.csv")
    m = (d.age.values > 0) * (d.age.values < 14)
    df = d.iloc[m]

    plt.clf()
    plt.plot(np.log(df.Jz.values**2), np.log(df.age.values), "k.")
    xs = np.linspace(min(df.Jz.values), max(df.Jz.values), 100)
    ys = .2*xs - (.9/.2)
    # plt.plot(xs, ys)
    plt.ylabel("Age")
    plt.xlabel("Jz2")
    plt.savefig("age_vs_Jz2")

    ages, dispersions, Ns = calc_bin_dispersion(df.age.values, df.Jz.values, 8)
    d_err = dispersions / (2 * Ns - 2)**.5

    plt.clf()
    plt.errorbar(ages - .5*(ages[1] - ages[0]), np.array(dispersions),
                 yerr=d_err, fmt="k.", capsize=0, ms=.1)
    plt.step(ages, dispersions, color="k")
    plt.xlabel("$\mathrm{Age~Gyr}$")
    plt.ylabel("$\sigma J_z~(\mathrm{Kpc~kms}^{-1})$")
    plt.savefig("linear_age_dispersion.pdf")

    m = np.log(df.age.values) > - 1
    lnages, dispersions, Ns = calc_bin_dispersion(np.log(df.age.values[m]),
                                      df.Jz.values[m], 8)
    d_err = dispersions / (2 * Ns - 2)**.5

    plt.clf()
    plt.errorbar(lnages - .5*(lnages[1] - lnages[0]), np.array(dispersions),
                 yerr=d_err, fmt="k.", capsize=0, ms=.1)
    plt.step(lnages, dispersions, color="k")
    plt.xlabel("$\ln(\mathrm{Age,~Gyr})$")
    plt.ylabel("$\sigma J_z~(\mathrm{Kpc~kms}^{-1})$")
    plt.xlim(-1, 2.6)
    xs = np.linspace(min(lnages), max(lnages), 100)
    ys = .2*xs + .9
    plt.plot(xs, ys)
    plt.savefig("log_age_dispersion.pdf")

    m = np.log(df.age.values) > - 1
    x, y = x_and_y(np.log(df.age.values[m]), df.Jz.values[m])

    plt.clf()
    plt.plot(x, y)
    plt.savefig("cont_age_dispersion.pdf")
