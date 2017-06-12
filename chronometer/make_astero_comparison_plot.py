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

plotpar = {'axes.labelsize': 18,
           'font.size': 10,
           'legend.fontsize': 18,
           'xtick.labelsize': 18,
           'ytick.labelsize': 18,
           'text.usetex': True}
plt.rcParams.update(plotpar)


def calc_iso_age(param_dict):
    """
    Calculate the isochronal ages samples.
    """
    model = StarModel(mist, **param_dict, use_emcee=True)
    model.fit()
    return model.samples.age_0_0


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

    # Load data.
    df = pd.read_csv(os.path.join(DATA_DIR, "vansaders_tgas_action.csv"))

    # Calculate isochronal ages.
    param_dict = {"Teff": (5770, 80), "logg": (4.44, .08), "feh": (.0, .01)}
    age, errp, errm, age_samples = calculate_isochronal_age(param_dict, "sun",
                                                            RESULTS_DIR)

    plt.clf()
    plt.hist(age_samples, 100)
    plt.savefig("test")

    age, errp, errm = meds(age_samples)
    print(age, errp, errm)

    # Load chronometer results.

    # Plot comparisons.
