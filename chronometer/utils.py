"""
Some utility functions used by chronometer.
"""

import os
import numpy as np
import pandas as pd
from isochrones import StarModel
from isochrones.mist import MIST_Isochrone

def replace_nans_with_inits(data):
    """
    Turn the data into reasonable initial values.
    params:
    ------
    data: (pandas.DataFrame)
        Data read from an input file - the colours and physical parameters of
        the stars.
    If no initial mass is provided, initialise with 1 solar mass.
    If no initial feh is provided, initialise with solar feh.
    If no initial Av is provided, initialise with 0.
    """
    data.mass.values[~np.isfinite(data.mass.values)] \
        = np.ones(len(data.mass.values[~np.isfinite(data.mass.values)]))
    data.feh.values[~np.isfinite(data.feh.values)] \
        = np.zeros(len(data.feh.values[~np.isfinite(data.feh.values)]))
    data.Av.values[~np.isfinite(data.Av.values)] \
        = np.zeros(len(data.Av.values[~np.isfinite(data.Av.values)]))
    data.parallax.values[~np.isfinite(data.parallax.values)] \
        = np.zeros(len(data.parallax.values[~np.isfinite(data.parallax.values)]))
    # data.parallax.values[~np.isfinite(data.parallax.values)] \
    #     = [None] * \
    #     (len(data.parallax.values[~np.isfinite(data.parallax.values)]))
    return data


def vk2teff(vk):
    """
    Convert V-K to Teff.
    From https://arxiv.org/pdf/astro-ph/9911367.pdf
    For 0.82 < VK < 3.29
    vk: (float)
        The V - K colour index.
    returns:
        Teff
    """
    a, b, c = 8686.22, -2441.65, 334.789
    return a + b*vk + c*vk**2


def make_param_dict(d, i):
    """
    Make a dictionary of the parameters.
    params:
    ------
    d: (pandas.DataFrame)
        The data.
    i: (int)
        The index of the star.
    """
    param_dict = {"V": (d.v.values[i], d.v_err.values[i]),
                  "Ks": (d.ks.values[i], d.ks_err.values[i]),
                  "J": (d.j.values[i], d.j_err.values[i]),
                  "H": (d.h.values[i], d.h_err.values[i]),
                  "K": (d.k.values[i], d.k_err.values[i]),
                  "Teff": (d.Teff.values[i], d.Teff_err.values[i]),
                  "logg": (d.logg.values[i], d.logg_err.values[i]),
                  "feh": (d.feh.values[i], d.feh_err.values[i]),
                  "parallax": (d.parallax.values[i],
                               d.parallax_err.values[i])}
    return param_dict


def transform_parameters(lnparams, indicator, all_params):
    if indicator == "gyro":
        if all_params:
            N = int((len(lnparams) - 3)/5)
            pars = lnparams[:3]
            ln_ages = lnparams[3+N:3+2*N]
        else:
            pars = lnparams[:3]
            ln_ages = lnparams[3:]
        return pars, ln_ages
    elif indicator == "iso":
        if all_params:
            p = lnparams[3:]*1
        else:
            p = lnparams*1
        N = int(len(p)/5)

        # Transform to linear space
        # mass, age, feh, distance, Av
        p[:N] = np.exp(p[:N])
        p[N:2*N] = np.log10(1e9*np.exp(p[N:2*N]))
        p[3*N:4*N] = np.exp(p[3*N:4*N])
        return p, N


def pars_and_mods(DATA_DIR):
    """
    Create the initial parameter array and the mod objects needed for
    isochrones.py from the data file provided.
    returns:
    -------
    p0: (array)
        The initial parameter values.
    mods: (list)
        A list of starmodel objects.
    """
    # The initial parameters
    gc = np.array([.7725, .601, .5189])
    d = pd.read_csv(os.path.join(DATA_DIR, "data_file.csv"))

    d = replace_nans_with_inits(d)
    p0 = np.concatenate((gc, np.log(d.mass.values),
                         np.log(d.age.values), d.feh.values,
                         np.log(1./d.parallax.values*1e3),
                         d.Av.values))

    # iso_lnlike preamble - make a list of 'mod' objects: one for each star.
    mist = MIST_Isochrone()
    mods = []
    for i in range(len(d.period.values)):
        param_dict = make_param_dict(d, i)

        # Remove missing parameters from the dict.
        param_dict = {k: param_dict[k] for k in param_dict if
                      np.isfinite(param_dict[k]).all()}
        mods.append(StarModel(mist, **param_dict))
    return p0, mods


def parameter_assignment(params, indicator):
    """
    Take the parameter array and split it into groups of iso parameters.
    params:
    ------
    params: (array)
        The parameter array.
    indicator: (string)
        Indicates whether the iso parameters or both iso and gyro parameters
        are being explored. Either 'iso' or 'both'.
    returns:
    -------
    N: (int)
        Number of stars.
    ln_mass, ln_age, etc: (arrays)
        The groups of parameters.
    """
    if indicator == "iso":
        N = int(len(params)/5.) # number of stars
        ln_mass = params[:N]  # parameter assignment
        ln_age = params[N:2*N]
        feh = params[2*N:3*N]
        ln_distance = params[3*N:4*N]
        Av = params[4*N:5*N]
    elif indicator == "both":
        N = int(len(params[3:])/5)  # number of stars
        ln_mass = params[3:3+N]  # parameter assignment
        ln_age = params[3+N:3+2*N]
        feh = params[3+2*N:3+3*N]
        ln_distance = params[3+3*N:3+4*N]
        Av = params[3+4*N:3+5*N]
    return N, ln_mass, ln_age, feh, ln_distance, Av
