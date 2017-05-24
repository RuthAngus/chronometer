"""
Some utility functions used by chronometer.
"""

import os
import numpy as np
import pandas as pd
from isochrones import StarModel
from isochrones.mist import MIST_Isochrone


def mask_column(d, col, val):
    """
    Turn all the NaNs into the val.
    params:
    ------
    d: (pandas.DataFrame)
        The data
    col: (array)
        The array to mask.
    val: (float)
        The value to convert the NaNs into.
    """
    m = np.isfinite(col)
    col[~m] = np.ones(len(col[~m]))*val
    return col


def replace_nans_with_inits(d):
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

    mass = mask_column(d, d.mass.values, 1.)
    d["mass"] = mass
    feh = mask_column(d, d.feh.values, 0.)
    d["feh"] = feh
    Av = mask_column(d, d.Av.values, 0.)
    d["Av"] = Av
    tgas_parallax = mask_column(d, d.tgas_parallax.values, 0.)
    d["tgas_parallax"] = tgas_parallax
    age = mask_column(d, d.age.values, np.log(2))
    d["age"] = age
    return d


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
    param_dict = {"J": (d.jmag.values[i], d.jmag_err.values[i]),
                  "H": (d.hmag.values[i], d.hmag_err.values[i]),
                  "K": (d.kmag.values[i], d.kmag_err.values[i]),
                  "G": (d.tgas_phot_g_mean_mag.values[i],
                        d.tgas_phot_g_mean_flux_error.values[i]/
                        d.tgas_phot_g_mean_flux.values[i] *
                        d.tgas_phot_g_mean_mag.values[i]),
                  "Teff": (d.teff.values[i], d.teff_err1.values[i]),
                  "logg": (d.logg.values[i], d.logg_err1.values[i]),
                  "feh": (d.feh.values[i], d.feh_err1.values[i]),
                  "parallax": (d.tgas_parallax.values[i],
                               d.tgas_parallax_error.values[i])}  # FIXME: add more filters
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


def pars_and_mods(d, global_params):
    """
    Create the initial parameter array and the mod objects needed for
    isochrones.py from the data file provided.
    Also, perform some operations on the data: merge multiple parallax columns
    and covert teffs into B-V.
    returns:
    -------
    p0: (array)
        The initial parameter values.
    mods: (list)
        A list of starmodel objects.
    """

    d = replace_nans_with_inits(d)
    p0 = np.concatenate((global_params, np.log(d.mass.values),
                         np.log(d.age.values), d.feh.values,
                         np.log(1./d.tgas_parallax.values*1e3),
                         d.Av.values))

    # iso_lnlike preamble - make a list of 'mod' objects: one for each star.
    mist = MIST_Isochrone()
    mods = []
    for i in range(len(d.prot.values)):
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
