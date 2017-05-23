"""
This script is for generating the input data file.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def create_and_save_df(*args):
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
    df.to_csv(os.path.join(DATA_DIR, "data.csv"))


if __name__ == "__main__":
    DATA_DIR = "/Users/ruthangus/projects/chronometer/chronometer/data"
    sun = pd.read_csv(os.path.join(DATA_DIR, "sun.csv"))
    NGC2019 = pd.read_csv(os.path.join(DATA_DIR, "NGC2019.csv"))
    tgas = pd.read_csv(os.path.join(DATA_DIR, "kic_tgas.csv"))
    tgas = tgas.iloc[:2, :]

    # create_and_save_df(sun, NGC2019)
    create_and_save_df(sun, NGC2019, tgas)
