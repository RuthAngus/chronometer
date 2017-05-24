"""
This script is for generating the input data file.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from calculate_actions import convert_coords_to_actions, save_pd_file
import teff_bv as tbv


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


if __name__ == "__main__":
    DATA_DIR = "/Users/ruthangus/projects/chronometer/chronometer/data"
    sun = pd.read_csv(os.path.join(DATA_DIR, "sun.csv"))
    NGC2019 = pd.read_csv(os.path.join(DATA_DIR, "NGC2019.csv"))
    tgas = pd.read_csv(os.path.join(DATA_DIR, "kic_tgas.csv"))
    tgas = tgas.iloc[:2, :]

    # create_and_save_df(sun, NGC2019)
    fn = "data.csv"
    create_and_save_df(fn, sun, NGC2019, tgas)
    df = pd.read_csv(os.path.join(DATA_DIR, fn))

    # Calculate B-Vs
    bv = tbv.teff2bv(df.teff.values, df.logg.values, df.feh.values)
    df["bv"] = bv

    # Calculate actions
    action_array = convert_coords_to_actions(df)
    save_pd_file(df, os.path.join(DATA_DIR, "action_data.csv"), action_array)
