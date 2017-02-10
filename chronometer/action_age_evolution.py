# # The Action-age relation

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from galpy.potential import MWPotential2014
from galpy.actionAngle import actionAngleStaeckel, actionAngleAdiabatic
from galpy.df import quasiisothermaldf


def calc_dispersion(time, sz0, t1, tm, beta, R0, Rc, Rsz):
    return sz0 * ((time + t1)/(tm + t1))**beta * np.exp((R0 - Rc)/Rsz)


if __name__ == "__main__":
    time = np.linspace(0, 14, 100)
    sz0 = 50.
    sr0 = 50.
    t1 = .1
    tm = 10.
    beta = .33
    R0 = 1.
    Rc = 1.
    hsz = 9.
    hsr = 9.

    sigma_z = calc_dispersion(time, sz0, t1, tm, beta, R0, Rc, hsz)
    sigma_r = calc_dispersion(time, sr0, t1, tm, beta, R0, Rc, hsr)

    plt.clf()
    plt.plot(time, sigma_z)
    plt.savefig("test")

    # Bovy example
    # aA = actionAngleAdiabatic(pot=MWPotential2014, c=True)
    # hr = 1/3.  # radial scale length
    # sr = .2  # radial velocity dispersion at the solar radius (time dependent)
    # sz = .1  # vertical velocity dispersion at the solar radius (time dependent)
    # hsr = 1.  # radial-velocity-dispersion scale length
    # hsz = 1.  # vertial-velocity-dispersion scale length

    aA = actionAngleStaeckel(pot=MWPotential2014, delta=0.45, c=True)
    solar_radius = 8.
    hr = 2.68/solar_radius
    # sr = 34.
    # sz = 25.1

    meanjzs = np.ones(len(sigma_z))
    for i, sz in enumerate(sigma_z):
        qdf = quasiisothermaldf(hr, sigma_r[i], sz, hsr, hsz,
                                pot=MWPotential2014, aA=aA, cutcounter=True)
        meanjzs.append(qdf.meanjz(1., .125))
    print(qdf.estimate_hr(1.))
    print(qdf.meanvz(1., 0.))
    print(qdf.meanjz(1., .125))
