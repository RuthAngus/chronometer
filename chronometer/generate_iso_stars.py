"""
This code generates the properties of fake stars.
"""

import numpy as np
import pandas as pd
from isochrones.mist import MIST_Isochrone
mist = MIST_Isochrone()


def age2period(par, age, bv):
    a, b, c, n = par
    return (age*1e3)**n * a * (bv - c)**b


def get_colors(mass, age, feh, distance, AV):
    return mist.mag['B'](mass, age, feh, distance, AV), \
        mist.mag['V'](mass, age, feh, distance, AV), \
        mist.mag['J'](mass, age, feh, distance, AV),\
        mist.mag['H'](mass, age, feh, distance, AV), \
        mist.mag['K'](mass, age, feh, distance, AV)


def star_colors(masses, ages, fehs, distances, Avs):
    B, V, J, H, K = [], [], [], [], []
    for i in range(len(masses)):
        b, v, j, h, k = get_colors(masses[i], ages[i], fehs[i], distances[i],
                                   Avs[i])
        B.append(b)
        V.append(v)
        J.append(j)
        H.append(h)
        K.append(k)
    return np.array(B), np.array(V), np.array(J), np.array(H), np.array(K)


if __name__ == "__main__":
    par = np.array([.7725, .60, .4, .5189])

    masses = np.array([.9, .95, 1., 1.05, 1.1])
    ages = np.array([np.log10(3.5e9), np.log10(6.5e9), np.log10(1e9),
                     np.log10(10e9), np.log10(4.5e9)])
    print(ages, "ages")
    fehs = np.array([-.2, -.05, 0., .1, -.1])
    distances = np.array([300, 200, 500, 400, 100])
    Avs = np.array([.2, .1, 0., .05, .15])

    B, V, J, H, K = star_colors(masses, ages, fehs, distances, Avs)
    periods = age2period(par, (10**ages)*1e-9, B-V)
    errs = [np.ones(len(B))*.01 for i in range(5)]

    dictionary = pd.DataFrame({"bv": B-V, "jmag": J, "jmag_err": errs[0],
                               "kmag": K, "kmag_err": errs[1], "hmag": H,
                               "hmag_err": errs[2], "prot": periods,
                               "prot_err": periods*.1, "tgas_parallax":
                               (1./distances)*1e3, "tgas_parallax_error":
                               np.ones(len(B))})

    dictionary.to_csv("data/fake_data.csv")
