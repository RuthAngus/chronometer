"""
Show that variance = x^2
"""

import numpy as np
import matplotlib.pyplot as plt

true_var = 5
N = 100000

xs = np.random.randn(N) * true_var**.5

var_xs = xs**2

plt.clf()
plt.hist(var_xs, 100)
plt.axvline(true_var, color="r", alpha=.5)
plt.axvline(np.median(true_var), color="g", alpha=.5, ls="--")
plt.savefig("var_test")
