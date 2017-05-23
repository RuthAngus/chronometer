"""
Take the parallax and proper motions of a TGAS star and calculate Jz.
Append Jz to a csv file.
"""

import numpy as np
import pandas as pd
from actions import calc_actions
import os


def load_pd_file(fn):
    df = pd.read_csv(fn)
    return df


def zero_fill(array):
    """
    Replace nans with zeros
    """
    m = np.isfinite(array)
    array[~m] = np.zeros(len(array[~m]))
    return array


def convert_coords_to_actions(df):
    """
    Take an existing pandas dataframe containing plx and pm info.
    Convert data to actions.
    """

    # Load info from dataframe and fill with zeros where unavailable.
    ra_deg, ra_deg_err = zero_fill(df.tgas_ra), zero_fill(df.tgas_ra_error)
    dec_deg, dec_deg_err = zero_fill(df.tgas_dec), zero_fill(df.tgas_dec_error)
    ra_dec_corr = zero_fill(df.tgas_ra_dec_corr)
    d_kpc = 1./df.tgas_parallax
    d_kpc_err = df.tgas_parallax_error/df.tgas_parallax * d_kpc
    d_kpc, d_kpc_err = zero_fill(d_kpc), zero_fill(d_kpc_err)
    pm_ra_masyr = zero_fill(df.tgas_pmra)
    pm_ra_masyr_err = zero_fill(df.tgas_pmra_error)
    pm_dec_masyr = zero_fill(df.tgas_pmdec)
    pm_dec_masyr_err = zero_fill(df.tgas_pmdec_error)
    # FIXME: add correlations

    # Look for RVs. If none exist, use zeros for now.
    try:
        v_los_kms = df.rv
        v_los_kms_err = df.rv_err
    except:
        v_los_kms = np.zeros(len(ra_deg))
        v_los_kms_err = np.zeros(len(ra_deg))


    # Calculate actions and errors.
    print(np.shape(ra_deg), np.shape(dec_deg), np.shape(d_kpc),
          np.shape(pm_ra_masyr), np.shape(pm_dec_masyr), np.shape(v_los_kms))
    print(ra_deg[0], dec_deg[0], d_kpc[0], pm_ra_masyr[0], pm_dec_masyr[0],
          v_los_kms[0])
    jr, lz, jz = calc_actions(ra_deg[0], dec_deg[0], d_kpc[0], pm_ra_masyr[0],
                              pm_dec_masyr[0], v_los_kms[0])
    jr_err, lz_err, jz_err = [np.zeros(len(jr)) for i in range(3)]
    print(jz)
    assert 0
    return jr, jr_err, lz, lz_err, jz, jz_err


def save_pd_file(df, fn, jr, jr_err, lz, lz_err, jz, jz_err):
    """
    Add the actions to the df and save it under a different name.
    """
    df["Jr"] = jr
    df["Jr_err"] = jr_err
    df["Lz"] = lz
    df["Lz_err"] = lz_err
    df["Jz"] = jz
    df["Jz_err"] = jz_err
    df.to_csv(fn)


if __name__ == "__main__":
    DIR = "/Users/ruthangus/projects/chronometer/chronometer/data"
    df = load_pd_file(os.path.join(DIR, "data.csv"))
    jr, jr_err, lz, lz_err, jz, jz_err = convert_coords_to_actions(df)
    print(jz)
    assert 0
    save_pd_file(df, os.path.join(DIR, "action_data.csv"), action_list)
