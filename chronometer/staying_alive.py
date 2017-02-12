import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import scipy.misc as spm
import corner


#!/usr/bin/env python
# Author:      Flávio Codeço Coelho
# License:     GPL

from math import *
import numpy as np
from matplotlib.pylab import *

# def sdnorm(z):
#     """
#     Standard normal pdf (Probability Density Function)
#     """
#     return exp(-z*z/2.)/sqrt(2*pi)

# n = 10000
# alpha = 1
# x = 0.
# vec = []
# vec.append(x)
# innov = np.random.uniform(-alpha,alpha,n) #random inovation, uniform proposal distribution
# for i in range(n):
#     can = x + innov[i] #candidate
#     aprob = min([1.,sdnorm(can)/sdnorm(x)]) #acceptance probability
#     u = uniform(0,1)
#     if u < aprob:
#         x = can
#         vec.append(x)

# #plotting the results:
# #theoretical curve
# x = arange(-3,3,.1)
# y = sdnorm(x)
# subplot(211)
# title('Metropolis-Hastings')
# plot(vec)
# subplot(212)

# hist(vec, bins=30,normed=1)
# plot(x,y,'ro')
# ylabel('Frequency')
# xlabel('x')
# legend(('PDF','Samples'))
# show()

def q(x, y):
    g1 = mlab.bivariate_normal(x, y, 1.0, 1.0, -1, -1, -0.8)
    g2 = mlab.bivariate_normal(x, y, 1.5, 0.8, 1, 2, 0.6)
    return 0.6*g1+28.4*g2/(0.6+28.4)


def model(par, x):
    return par[0] + par[1]*x


def lnlike(par, x, y_obs, y_err):
    y_mod = model(par, x)
    return sum(-.5*((y_mod - y_obs)/y_err)**2)


def metropolis_hastings(x, y, yerr, r, N, s, t):
    """
    params:
    -------
    r: (tuple or list or array)
        The parameters/position in probability space.
    N: (int)
        Number of samples.
    s: (int)
        Thinning index

    """
    print("inits = ", r)
    p = np.exp(lnlike(r, x, y, yerr))  # Initial probability.
    samples = []
    for i in np.arange(N):
        rn = r + np.random.normal(size=2)*t  # Take a step in parameter space.
        pn = np.exp(lnlike(r, x, y, yerr))  # calculate the probability.
        if pn >= p:  # If the new probability is greater than the previous p
            p = pn  # The probability is now the new one.
            r = rn  # adopt the new parameters.
        else:
            u = np.random.rand()  # Otherwise, random number between 0 and 1.
            if u < pn/p:  # if less than the ratio of new p to old p:
                p = pn  # prob is now new prob.
                r = rn  # adopt the new parameters.
        if i % s == 0:  # If this is the 10*th step, append sample (thinning)
            samples.append(r)
    return np.array(samples), r


if __name__ == "__main__":

    # Truth
    x = np.arange(0, 10, .1)
    err = 2.
    yerr = np.ones_like(x) * err
    y = .7 + 2.5*x + np.random.randn(len(x))*err

    plt.clf()
    plt.errorbar(x, y, yerr=yerr, fmt="k.")
    plt.savefig("data")

    # First, try random sampling.
    m = np.random.randn(1000) + 2.5
    c = np.random.randn(1000) + .5
    Lm = np.array([np.exp(lnlike([.7, m[i]], x, y, yerr)) for i in range(len(m))])
    Lc = np.array([np.exp(lnlike([c[i], 2.5], x, y, yerr)) for i in range(len(m))])

    plt.clf()
    plt.plot(m, Lm, "k.")
    plt.axvline(2.5)
    plt.xlabel("m")
    plt.ylabel("likelihood")
    plt.savefig("corner_m")
    plt.clf()
    plt.plot(c, Lc, "k.")
    plt.axvline(.7)
    plt.xlabel("c")
    plt.ylabel("likelihood")
    plt.savefig("corner_c")

    lc, lm = Lc == max(Lc), Lm == max(Lm)
    print(m[lm], c[lc])


    '''Metropolis Hastings'''
    N = 1000000  # N samples
    s = 1  #
    r = [.5, 2.5]  # initialisation
    t = .0001

    samples, r = metropolis_hastings(x, y, yerr, r, N, s, t)

    print(np.shape(samples))
    plt.clf()
    plt.hist(samples[:, 0], 100)
    plt.axvline(.7, color="HotPink")
    print(samples[:100, 0])
    plt.savefig("corner")
