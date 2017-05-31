"""
This script is for generating the input data file.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from calculate_actions import convert_coords_to_actions, save_pd_file
import teff_bv as tbv
from gyro_vs_iso_plot import calculate_gyrochronal_ages


def create_and_save_df(fn, *args):
    """
    Create the pandas dataframe, append dictionaries to it and save it as a
    csv file.
    input:
    -----
    args:
        Varying numbers of dictionaries containing data to be added to the
        dataframe.
    """
    frames = [i for i in args]
    df = pd.concat(frames, axis=0)
    df.to_csv(os.path.join(DATA_DIR, fn))


def add_actions_and_colours(df, fn):
    """
    Given a pandas dataframe (of the kepler and TGAS intersection), calculate
    B-V colours and actions and add these properties to the dataset.
    fn is the name of the file to save as, including the path.
    """

    # Calculate B-Vs
    bv = tbv.teff2bv(df.teff.values, df.logg.values, df.feh.values)
    df["bv"] = bv

    par = [.7725, .601, .4, .5189]
    age = calculate_gyrochronal_ages(par, df.prot.values, bv)
    print(age)

    # Calculate actions
    beta = 350.
    # R_kpc, R_kpc_err, phi_rad, phi_rad_err, z_kpc, z_kpc_err, vR_kms, \
    #     vR_kms_err, vT_kms, vT_kms_err, vz_kms, vz_kms_err, Jr, Jr_err, Lz, \
    #     Lz_err, Jz_err =
    stuff = np.array([np.zeros(len(df.teff.values)) for i in range(16)])
    print((beta * age)**.5)
    Jz2 = np.abs(np.random.randn(len(age)) * beta * age)
    Jz = Jz2**.5
    Jz_err = np.zeros(len(df.teff.values))
    action_array = np.vstack((stuff, Jz, Jz_err))
    # print(action_array)
    save_pd_file(df, fn, action_array)

    print(df["prot"])
    # print(df["tgas_parallax"])
    print(df["Jz"])

if __name__ == "__main__":
    DATA_DIR = "/Users/ruthangus/projects/chronometer/chronometer/data"

    # Load dataframes for data you want to combine
    sun = pd.read_csv(os.path.join(DATA_DIR, "sun.csv"))
    NGC2019 = pd.read_csv(os.path.join(DATA_DIR, "NGC2019.csv"))
    tgas = pd.read_csv(os.path.join(DATA_DIR, "kplr_tgas_periods.csv"))
    tgas = tgas.iloc[:2, :]

    # Join data files together.
    fn = "fake.csv"
    create_and_save_df(fn, sun, NGC2019, tgas)

    # Calculate B-V colour and actions and append to data file.
    df = pd.read_csv(os.path.join(DATA_DIR, fn))
    add_actions_and_colours(df, os.path.join(DATA_DIR,
                                             "fake_action_data.csv"))
