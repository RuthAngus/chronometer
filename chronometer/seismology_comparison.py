# Compare the bulk parameters of Chaplin et al with those re-derived using
# isochrones.py

import numpy as np
import matplotlib.pyplot as plt
from isochrones.dartmouth import Dartmouth_Isochrone
from isochrones.mist import MIST_Isochrone
from isochrones import StarModel


dar = Dartmouth_Isochrone()
mist = MIST_Isochrone()

mass, age, feh = (1.01, 9.71, 0.01)
print(dar.logg(mass, age, feh), mist.logg(mass, age, feh))
print(dar.Teff(mass, age, feh), mist.Teff(mass, age, feh))

mod = StarModel(mist, Teff=(5700,100), logg=(4.5,0.1), feh=(0.0,0.1))
mod.fit(basename='spec_demo')
