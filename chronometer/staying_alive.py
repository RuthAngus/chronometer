import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab


def q(x, y):
    g1 = mlab.bivariate_normal(x, y, 1.0, 1.0, -1, -1, -0.8)
    g2 = mlab.bivariate_normal(x, y, 1.5, 0.8, 1, 2, 0.6)
    return 0.6*g1+28.4*g2/(0.6+28.4)


def metropolis_hastings(N, r, s):
    """
    params:
    -------
    N: (int)
        Number of samples.
    r: (tuple or list or array)
        The parameters/position in probability space.
    s: (int)
        Thinning index

    """

    p = q(r[0], r[1])  # Initial probability.
    samples = []
    for i in np.arange(N):
        rn = r + np.random.normal(size=2)  # Take a step in parameter space.
        pn = q(rn[0], rn[1])  # calculate the probability.
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
    return np.array(samples)



if __name__ == "__main__":

    '''Metropolis Hastings'''
    N = 100000  # N samples
    s = 10  #
    r = np.zeros(2)  # initialisation

    samples = metropolis_hastings(N, r, s)

    plt.clf()
    plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5, s=1)

    '''Plot target distribution'''
    dx = 0.01
    x = np.arange(np.min(samples), np.max(samples), dx)
    y = np.arange(np.min(samples), np.max(samples), dx)
    X, Y = np.meshgrid(x, y)
    Z = q(X, Y)
    CS = plt.contour(X, Y, Z, 10)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.savefig("test")
