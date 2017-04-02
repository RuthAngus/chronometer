"""
The age models.
"""

import numpy as np

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
    return a*(np.exp(ln_age)*1e3)**n * (bv - .4)**b
