# Join astero, KIC and TGAS

import os
import numpy as np
import pandas as pd

DIR = "/Users/ruthangus/projects/chronometer/chronometer/data"

# Load asteroseismic data.
c1 = pd.read_csv(os.path.join(DIR, "table1_seismic.csv"))
c2 = pd.read_csv(os.path.join(DIR, "table2_seismic_bruntt.csv"))
c4 = pd.read_csv(os.path.join(DIR, "table4_stellar_SDSS.csv"))
c5 = pd.read_csv(os.path.join(DIR, "table5_stellar_IRFM.csv"))
c6 = pd.read_csv(os.path.join(DIR, "table6_stellar_bruntt.csv"))
metcalfe = pd.read_csv(os.path.join(DIR, "metcalfe.csv"))
vansaders = pd.read_csv(os.path.join(DIR, "vanSaders.csv"))

# First replace c1 with c2 values
c1c2 = pd.merge(c1, c2, on="kepid", how="outer", suffixes=["", "_bruntt"])

# Replace c5 with c6 values

# Replace c4 with c5c6 values

# Join c4c5c6 with c1c2

# Replace c values with metcalfe, vanSaders and silva-aguirre values.

# load KIC/TGAS
# kic_tgas = pd.read_csv(os.path.join(DIR "kic_tgas.csv"))
