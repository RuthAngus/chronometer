"""
Test the emcee functions in chronometer.py
"""

import os
import time
import numpy as np
import pandas as pd
from emcee_chronometer import emcee_lnprob, emcee_lnprior, get_n_things
from utils import pars_and_mods


def make_args(mods, params, d):
    N, ngyro, nglob, nind = get_n_things(mods, params)

    # Construct parameter indices for the different parameter sets.
    par_inds = np.arange(len(params))  # All
    g_par_inds = np.concatenate((par_inds[:ngyro],
                                 par_inds[nglob+N:nglob+2*N]))
    m = (d.bv.values > .4) * (np.isfinite(d.prot.values))
    g_par_inds_mask = np.concatenate((g_par_inds[:3], g_par_inds[3:][m]))

    # A list of kinematic indices.
    kin_inds = list(range(nglob+N, nglob+N+nind))
    kin_inds.insert(0, ngyro)

    return [mods, d.prot.values, d.prot_err.values, d.bv.values,
            d.bv_err.values, d.Jz, d.Jz_err, N, ngyro, nglob, nind,
            g_par_inds_mask, kin_inds, m]


def test_emcee_lnprob():
    DATA_DIR = "/Users/ruthangus/projects/chronometer/chronometer/data"
    d = pd.read_csv(os.path.join(DATA_DIR, "action_data.csv"))
    global_params = np.array([.7725, .601, .5189, np.log(350.)])
    params, mods = pars_and_mods(d, global_params)
    emcee_args = make_args(mods, params, d)
    start = time.time()
    print(emcee_lnprob(params, *emcee_args))
    end = time.time()
    print("time taken = ", end - start, "seconds")
    start = time.time()
    print(emcee_lnprob(params, *emcee_args))
    end = time.time()
    print("time taken = ", end - start, "seconds")


def test_emcee_lnprior():
    DATA_DIR = "/Users/ruthangus/projects/chronometer/chronometer/data"
    d = pd.read_csv(os.path.join(DATA_DIR, "action_data.csv"))
    global_params = np.array([.7725, .601, .5189, np.log(350.)])
    params, mods = pars_and_mods(d, global_params)
    emcee_args = make_args(mods, params, d)
    print(emcee_lnprior(params, *emcee_args))

if __name__ == "__main__":
    test_emcee_lnprior()
    test_emcee_lnprob()
