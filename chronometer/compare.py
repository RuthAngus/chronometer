"""
Compare the properties injected to the properties recovered.
Particularly the Ages.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py


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


def make_comparison_plot(true, recovered, xlabel, ylabel, fn):
    """
    Compare the true property with the injected property.
    """


if __name__ == "__main__":

    RESULTS_DIR = "/Users/ruthangus/projects/chronometer/chronometer/MH"
    DATA_DIR = "/Users/ruthangus/projects/chronometer/chronometer/data"

    # Load truths
    df = pd.read_csv(os.path.join(DATA_DIR, "fake_data.csv"))
    true_ages = df.age.values

    # Load samples
    with h5py.File(os.path.join(RESULTS_DIR, "combined_samples.h5"),
                   "r") as f:
        samples = f["samples"][...]

    npar = np.shape(samples)[1]
    N = int((npar - 4)/5)
    nglob = 4

    print(N, "stars")

    recovered_age_samples = samples[:, nglob+N:nglob+2*N]
    print(np.shape(recovered_age_samples))

    samples = np.zeros((1000, 5))
    samples[:, 0] = np.log(np.random.randn(1000)*1 + 100)
    samples[:, 1] = np.log(np.random.randn(1000)*2 + 101)
    samples[:, 2] = np.log(np.random.randn(1000)*3 + 102)
    samples[:, 3] = np.log(np.random.randn(1000)*4 + 103)
    samples[:, 4] = np.log(np.random.randn(1000)*5 + 104)

    meds, errm, errp = get_stats_from_samples(np.exp(samples))
    print(meds, errm, errp)
