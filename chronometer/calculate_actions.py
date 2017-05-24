"""
Take the parallax and proper motions of a TGAS star and calculate Jz.
Append Jz to a csv file.
Save file as action_data.csv
"""

import numpy as np
import pandas as pd
from actions import action
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
    R_kpc, phi_rad, z_kpc, vR_kms, vT_kms, vz_kms, jr, lz, jz = \
        [np.zeros(len(ra_deg)) for i in range(9)]
    jr_err, lz_err, jz_err, R_kpc_err, phi_rad_err, z_kpc_err, vR_kms_err, \
        vT_kms_err, vz_kms_err = [np.zeros(len(jr)) for i in range(9)]
    for i in range(len(ra_deg)):
        R_kpc[i], phi_rad[i], z_kpc[i], vR_kms[i], vT_kms[i], vz_kms[i], \
            jr[i], lz[i], jz[i] = action(ra_deg[i], dec_deg[i], d_kpc[i],
                                         pm_ra_masyr[i], pm_dec_masyr[i],
                                         v_los_kms[i])
    m = (ra_deg == 0.) * (dec_deg == 0.) * (pm_ra_masyr == 0.) * \
        (pm_dec_masyr == 0.)
    action_array = np.array([R_kpc, R_kpc_err, phi_rad, phi_rad_err, z_kpc,
                             z_kpc_err, vR_kms, vR_kms_err, vT_kms,
                             vT_kms_err, vz_kms, vz_kms_err, jr, jr_err, lz,
                             lz_err, jz, jz_err])
    for i, arr in enumerate(action_array):  # Mute non-values
        action_array[i][m] = np.ones(len(action_array[i][m])) * np.nan
    return action_array


def save_pd_file(df, fn, array):
    """
    Add the actions to the df and save it under a different name.
    """
    df["R_kpc"], df["R_kpc_err"], df["phi_rad"], df["phi_rad_err"], \
        df["z_kpc"], df["z_kpc_err"], df["vR_kms"], df["vR_kms_err"], \
        df["vT_kms"], df["vT_kms_err"], df["vz_kms"], df["vz_kms_err"], \
        df["Jr"], df["Jr_err"], df["Lz"], df["Lz_err"], df["Jz"], \
        df["Jz_err"] = array
    print(df["R_kpc"])
    df.to_csv(fn)


if __name__ == "__main__":
    DIR = "/Users/ruthangus/projects/chronometer/chronometer/data"
    df = load_pd_file(os.path.join(DIR, "data.csv"))
    action_array = convert_coords_to_actions(df)
    save_pd_file(df, os.path.join(DIR, "action_data.csv"), action_array)
