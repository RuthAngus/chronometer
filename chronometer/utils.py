"""
Some utility functions used by chronometer.
"""

import numpy as np

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
    # data.parallax.values[~np.isfinite(data.parallax.values)] \
    #     = [None] * \
    #     (len(data.parallax.values[~np.isfinite(data.parallax.values)]))
    return data


def vk2teff(vk):
    """
    Convert V-K to Teff.
    From https://arxiv.org/pdf/astro-ph/9911367.pdf
    For 0.82 < VK < 3.29
    """
    a, b, c = 8686.22, -2441.65, 334.789
    return a + b*vk + c*vk**2


def make_param_dict(d, i):
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
