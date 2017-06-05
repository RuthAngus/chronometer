"""
The age models.
"""

import numpy as np
import pandas as pd


def gc_model(params, ln_age, bv):
    """
    Given a B-V colour and an age predict a rotation period.
    Returns log(age) in Myr.
    parameters:
    ----------
    params: (array)
        The array of age and (log) gyro parameters, a, b and n.
    data: (array)
        A an array containing colour.
    """
    a, b, n = params
    # a, b, n = .7725, .601, .5189
    return a*(np.exp(ln_age)*1e3)**n * (bv - .4)**b


def gyro_model(params, bv):
    """
    Given a B-V colour and an age predict a rotation period.
    Returns log(age) in Myr.
    parameters:
    ----------
    params: (array)
        The array of age and (log) gyro parameters, a, b and n.
    data: (array)
        A an array containing colour.
    """
    a, b, n = params[:3]
    N = len(bv)
    ln_age = params[3:3+N]
    return a*(np.exp(ln_age)*1e3)**n * (bv - .4)**b


def pure_gyro_model(params, ln_age, bv):
    """
    Given a B-V colour and an age predict a rotation period.
    parameters:
    ----------
    params: (array)
        The array of age and (log) gyro parameters, a, b and n.
    data: (array)
        A an array containing colour.
    """
    a, b, n = params
    return a*(np.exp(ln_age)*1e3)**n * (bv - .4)**b


def pure_action_age(par, ln_age, Jz, Jz_err):
    """
    Given a vertical action, calculate an age.
    Vertical action dispersion increases with time.
    Vertical action is drawn from a Normal distribution with zero mean and
    dispersion that is a function of time.
    """
    beta, ages = np.exp(par), np.exp(ln_age)
    if beta > 0:
        return np.sum(-.5*(Jz**2/(beta*ages + Jz_err**2)) - \
            .5*np.log(2*np.pi*(beta*ages + Jz_err**2)))
    else:
        return -np.inf


def action_age(par, Jz, Jz_err):
    """
    Given a vertical action, calculate an age.
    Vertical action dispersion increases with time.
    Vertical action is drawn from a Normal distribution with zero mean and
    dispersion that is a function of time.
    """
    t = pd.read_csv("truths.txt")
    beta, ages = np.exp(par[0]), np.exp(par[1:])
    ages = t.ages.values
    if beta > 0:
        return np.sum(-.5*(Jz**2/(beta*ages + Jz_err**2)) - \
            .5*np.log(2*np.pi*(beta*ages + Jz_err**2)))
    else:
        return -np.inf
