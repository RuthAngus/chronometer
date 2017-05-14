"""
Test the models in models.py
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from models import action_age
import emcee

plotpar = {'axes.labelsize': 18,
           'font.size': 10,
           'legend.fontsize': 18,
           'xtick.labelsize': 18,
           'ytick.labelsize': 18,
           'text.usetex': True}
plt.rcParams.update(plotpar)


def jz_lnprior(beta):
    if 0 < beta < 1e20:
        return 0.
    else:
        return -np.inf


def lnprob(par, t, Jz, Jz_err):
    return action_age(par, t, Jz, Jz_err) + jz_lnprior(par)


def test_action_age_model():
    DATA_DIR = "/Users/ruthangus/granola/granola/data"
    d = pd.read_csv("ages_and_actions.csv")
    m = (d.age.values > 0) * (d.age.values < 14)
    df = d.iloc[m]
    jz, t = df.Jz.values, df.age.values
    jz_err = np.zeros(len(jz))
    print(action_age(1e4, t, jz, np.zeros(len(jz))))
    args = [t, jz, jz_err]
    p0 = 1e4
    init = p0*1

    nwalkers, nsteps, ndim = 64, 1000, 1
    p0 = [1e-4*np.random.rand(ndim) + p0 for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=args)
    print("burning in...")
    pos, _, _ = sampler.run_mcmc(p0, 500)
    sampler.reset()
    print("production run...")
    sampler.run_mcmc(pos, nsteps)

    flat = np.reshape(sampler.chain, (nwalkers*nsteps, ndim))

    plt.clf()
    plt.hist(flat[:, 0])
    plt.savefig("hist_test")

    result = np.median(flat[:, 0])

    return init, result


def plot_model(t, jz, init, result):
    xs = np.linspace(min(t), max(t), 100)
    plt.clf()
    # plt.loglog()
    plt.plot(t, jz**2, "k.")
    plt.plot(xs, init**.5*xs)
    plt.plot(xs, result**.5*xs)
    plt.xlabel("$\mathrm{Time~(Gyr)}$")
    plt.ylabel("$J_z^2$")
    plt.subplots_adjust(left=.15, bottom=.15)
    plt.savefig("test")


def make_fake_data():
    x1 = np.linspace(.1, 50, 500)
    x2 = np.linspace(50, 100, 500)
    x3 = np.linspace(100, 150, 500)
    x4 = np.linspace(150, 200, 500)
    x5 = np.linspace(200, 250, 500)
    y1 = ((np.random.randn(len(x1)) * 50)**2)**.5
    y2 = ((np.random.randn(len(x2)) * 100)**2)**.5
    y3 = ((np.random.randn(len(x2)) * 150)**2)**.5
    y4 = ((np.random.randn(len(x2)) * 200)**2)**.5
    y5 = ((np.random.randn(len(x2)) * 250)**2)**.5

    xs = np.linspace(0, 250, 100)
    ys = 1 * xs
    # xs = [50, 100, 150, 200, 250]
    # ys = [50**.5, 100**.5, 150**.5, 200**.5, 250**.5]

    plt.clf()
    plt.plot(x1, y1, ".")
    plt.plot(x2, y2, ".")
    plt.plot(x3, y3, ".")
    plt.plot(x4, y4, ".")
    plt.plot(x5, y5, ".")
    plt.plot(xs, ys)
    plt.savefig("fake_data")

    return np.concatenate((x1, x2, x3, x4, x5)), \
        np.concatenate((y1, y2, y3, y4, y5))**.5


def test_var_model(t, jz):
    jz_err = np.zeros(len(jz))

    args = [t, jz, jz_err]
    p0 = 1.
    init = p0*1

    nwalkers, nsteps, ndim = 64, 10000, 1
    p0 = [1e-4*np.random.rand(ndim) + p0 for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=args)
    print("burning in...")
    pos, _, _ = sampler.run_mcmc(p0, 500)
    sampler.reset()
    print("production run...")
    sampler.run_mcmc(pos, nsteps)

    flat = np.reshape(sampler.chain, (nwalkers*nsteps, ndim))

    plt.clf()
    plt.hist(flat[:, 0])
    plt.savefig("var_test")
    result = np.median(flat[:, 0])
    return init, result


if __name__ == "__main__":
    t, jz = make_fake_data()
    init, result = test_var_model(t, jz)
    plot_model(t, jz, init, result)
    assert 0
    init, result = test_action_age_model()
