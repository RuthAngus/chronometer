"""
Make a plot comparing the asteroseismic ages with isochronal ages and
chronometer ages.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import isochrones
from isochrones.mist import MIST_Isochrone
from isochrones import StarModel
from isochrones.mist import MIST_Isochrone
mist = MIST_Isochrone()

from gyro_vs_iso_plot import calculate_isochronal_age
from utils import pars_and_mods

plotpar = {'axes.labelsize': 26,
           'font.size': 10,
           'legend.fontsize': 18,
           'xtick.labelsize': 26,
           'ytick.labelsize': 26,
           'text.usetex': True}
plt.rcParams.update(plotpar)


def meds(samps):
    """
    Get medians and confidence intervals.
    """
    med = np.median(samps)
    errp = np.percentile(samps, 84) - med
    errm = med - np.percentile(samps, 16)
    return med, errp, errm


if __name__ == "__main__":

    DATA_DIR = os.path.join(os.getcwd(), "data")
    RESULTS_DIR = os.path.join(os.getcwd(), "test")

    N = 10

    # Load data.
    d = pd.read_csv(os.path.join(DATA_DIR, "vansaders_tgas_action.csv"))
    d = d.iloc[:N]

    param_dict = {"Teff": (5770, 80), "logg": (4.44, .08), "feh": (.0, .01)}
    age, errp, errm, age_samples = calculate_isochronal_age(param_dict, "sun",
                                                            RESULTS_DIR)
    print(age, errp, errm)
    print(meds(age_samples))

    # Generate the initial parameter array and the mods objects from the data
    # global_params = np.array([.7725, .601, .5189, np.log(350.)])  # a b n beta
    global_params = np.array([.4, .31, .55, np.log(350.)])  # a b n beta
    init_params, mods = pars_and_mods(d, global_params)

    # Calculate isochronal ages.
    ages, errps, errms = [np.zeros(N) for i in range(3)]
    for i in range(N):
        print(i, "of", N)
        _, _, _, age_samples = calculate_isochronal_age(mods[i], i,
                                                                RESULTS_DIR)
        age, errp, errm = meds(age_samples)
        print(age, errp, errm)
        ages[i], errps[i], errms[i] = age, errp, errm

        plt.clf()
        plt.hist(age_samples, 50)
        plt.savefig(os.path.join(RESULTS_DIR, "{}_hist".format(i)))

    # Load chronometer results.
    astero_ages, astero_age_errs = d.basta_age, d.basta_age_err

    # Plot comparisons.
    xs = np.linspace(0, 15, 100)
    plt.clf()
    plt.errorbar(astero_ages, ages, xerr=astero_age_errs, yerr=[errps, errms],
                 fmt="k.")
    plt.plot(xs, xs, color=".5", ls="--")
    plt.xlabel("$\mathrm{Asteroseismic~age~(Gyr)}$")
    plt.ylabel("$\mathrm{Isochronal~age~(Gyr)}$")
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.subplots_adjust(left=.15, bottom=.15)
    plt.savefig("figures/astero_vs_iso.pdf")
