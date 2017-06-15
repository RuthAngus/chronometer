"""
Compare the properties injected to the properties recovered.
Particularly the Ages.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py

plotpar = {'axes.labelsize': 18,
           'font.size': 10,
           'legend.fontsize': 18,
           'xtick.labelsize': 18,
           'ytick.labelsize': 18,
           'text.usetex': True}
plt.rcParams.update(plotpar)


def get_stats_from_samples(samples):
    """
    Take a 2d array of samples and produce medians and confidence intervals.
    """
    meds = np.array([np.median(samples[:, i]) for i in
                     range(np.shape(samples)[1])])
    lower = np.array([np.percentile(samples[:, i], 16) for i in
                        range(np.shape(samples)[1])])
    upper = np.array([np.percentile(samples[:, i], 84) for i in
                        range(np.shape(samples)[1])])
    errm, errp = meds - lower, upper - meds
    return meds, errm, errp


def make_comparison_plot(true, recovered, errp, errm, iso, ierrp, ierrm,
                         xlabel, ylabel, fn):
    """
    Compare the true property with the injected property.
    """
    xs = np.linspace(min(true), max(true))
    plt.clf()
    plt.errorbar(true, recovered, yerr=[errm, errp], fmt="k.")
    plt.errorbar(true, iso, yerr=[ierrm, ierrp], fmt="m.", alpha=.5)
    plt.plot(xs, xs, "--", color=".5")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.subplots_adjust(bottom=.15)
    plt.savefig(fn)
    print(np.median(errm), np.median(errp), np.median(ierrm),
          np.median(ierrp))
    print(np.mean([np.median(ierrm), np.median(ierrp)])
          /np.mean([np.median(errm), np.median(errp)]))


if __name__ == "__main__":

    cwd = os.getcwd()
    RESULTS_DIR = "/Users/ruthangus/projects/chronometer/chronometer/MH"
    DATA_DIR = "/Users/ruthangus/projects/chronometer/chronometer/data"

    # Load samples
    with h5py.File(os.path.join(RESULTS_DIR, "combined_samples.h5"),
                   "r") as f:
        samples = f["samples"][...]

    # Find N stars
    npar = np.shape(samples)[1]
    N = int((npar - 4)/5)
    nglob = 4
    print(N, "stars")

    # Load iso only samples
    with h5py.File(os.path.join(RESULTS_DIR, "combined_samples_iso_only.h5"),
                   "r") as f:
        iso_samples = f["samples"][...]

    # Calculate medians and errorbars
    recovered_age_samples = samples[:, nglob+N:nglob+2*N]
    meds, errm, errp = get_stats_from_samples(np.exp(recovered_age_samples))
    iso_age_samples = iso_samples[:, nglob+N:nglob+2*N]
    iso, ierrm, ierrp = get_stats_from_samples(np.exp(iso_age_samples))

    # Load truths
    df = pd.read_csv(os.path.join(DATA_DIR, "fake_data.csv"))
    true_ages = df.age.values[:N]

    # Make plot
    make_comparison_plot(true_ages, meds, errp, errm, iso, ierrp, ierrm,
                         "$\mathrm{True~age~(Gyr)}$",
                         "$\mathrm{Recovered~age~(Gyr)}$",
                         "compare_ages_{}".format(N))
