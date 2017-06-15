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


def add_actions_and_colours(df, fn):
    """
    Given a pandas dataframe (of the kepler and TGAS intersection), calculate
    B-V colours and actions and add these properties to the dataset.
    fn is the name of the file to save as, including the path.
    """

    # Calculate B-Vs
    bv = tbv.teff2bv(df.teff.values, df.logg.values, df.feh.values)
    bv_err = bv * .01
    df["bv"] = bv
    df["bv_err"] = bv_err

    # Calculate actions
    action_array = convert_coords_to_actions(df)
    save_pd_file(df, fn, action_array)

    print(df["prot"])
    print(df["tgas_parallax"])

if __name__ == "__main__":
    DATA_DIR = "/Users/ruthangus/projects/chronometer/chronometer/data"

    # Load dataframes for data you want to combine
    sun = pd.read_csv(os.path.join(DATA_DIR, "sun.csv"))
    NGC2019 = pd.read_csv(os.path.join(DATA_DIR, "NGC2019.csv"))
    tgas = pd.read_csv(os.path.join(DATA_DIR, "kplr_tgas_periods.csv"))
    tgas = tgas.iloc[:100, :]
    fn = "data.csv"

    # Join data files together.
    create_and_save_df(fn, sun, tgas)

    # Calculate B-V colour and actions and append to data file.
    df = pd.read_csv(os.path.join(DATA_DIR, fn))
    add_actions_and_colours(df, os.path.join(DATA_DIR, "action_data.csv"))
