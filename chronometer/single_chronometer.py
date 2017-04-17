"""
Assembling likelihood functions for Gyrochronology, isochrones, dynamics
and asteroseismology.
"""

import os

import time
import numpy as np
import matplotlib.pyplot as plt
import h5py

import emcee
import corner
from isochrones import StarModel
from isochrones.mist import MIST_Isochrone

import priors

plotpar = {'axes.labelsize': 20,
           'font.size': 20,
           'legend.fontsize': 20,
           'xtick.labelsize': 15,
           'ytick.labelsize': 15,
           'text.usetex': True}
plt.rcParams.update(plotpar)


def GC_model(params, bv):
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
    a, b, n, ln_age = params
    a, b, n = [.7725, .601, .5189]
    return a*(np.exp(ln_age)*1e3)**n * (bv - .4)**b


def gc_lnlike(params, period, bv):
    """
    Probability of age and model parameters given rotation period and colour.
    parameters:
    ----------
    params: (array)
        The array of log parameters: a, b, n, age.
    period: (tuple)
        The rotation period and period uncertainty in days
    bv: (tuple)
        The B-V colour and colour uncertainty.
    """
    model_periods = GC_model(params[:4], bv[0])
    return -.5*((period[0] - model_periods)/period[1])**2


def iso_lnlike(lnparams, mod):
    """
    Some isochronal likelihood function.
    parameters:
    ----------
    params: (array)
        The array of parameters: log(mass), log(age in Gyr), metallicity,
        log(distance), extinction.
    mod: (object)
        An isochrones.py starmodel object.
    """
    a, b, n, lnage, lnmass, feh, lnd, Av = lnparams
    params = np.array([np.exp(lnmass), np.log10(1e9*np.exp(lnage)), feh,
                       np.exp(lnd), Av])
    return mod.lnlike(params)


def lnprior(params):
    age_prior = np.log(priors.age_prior(np.log10(1e9*np.exp(params[3]))))
    feh_prior = np.log(priors.feh_prior(params[5]))
    distance_prior = np.log(priors.distance_prior(np.exp(params[6])))
    Av_prior = np.log(priors.AV_prior(params[7]))
    if -10 < params[0] < 10 and -10 < params[1] < 10 and \
            -10 < params[2] < 10 and -2 < params[4] < 3:
        return age_prior + feh_prior + distance_prior + Av_prior
    else:
        return -np.inf


def lnprob(params, mod, period, bv, gyro=True, iso=True):
    """
    The joint log-probability of age given gyro and iso parameters.
    mod: (list)
        list of pre-computed star model objects.
    gyro: (bool)
        If True, the gyro likelihood will be used.
    iso: (bool)
        If True, the iso likelihood will be used.
    """

    if gyro and iso:
        return iso_lnlike(params, mod) + gc_lnlike(params, period, bv) + \
            lnprior(params)
    elif gyro:
        return gc_lnlike(params, period, bv) + lnprior(params)
    elif iso:
        return iso_lnlike(params, mod) + lnprior(params)


def probtest(xs, i):
    lps = []
    p = params + 0
    for x in xs:
        p[i] = x
        lp = lnprob(p, mod, period, bv, gyro=True, iso=True)
        lps.append(lp)
    plt.clf()
    plt.plot(xs, lps)
    plt.xlabel("X")
    plt.ylabel("lnprob")
    plt.savefig(os.path.join(FIG_DIR, "probs_{}".format(i)))


def am(M, D):
    """
    apparent magnitude
    """
    return 5*np.log10(D) - 5 - M


def sun_demo(fname, gyro=True, iso=True, plot=False, spec=False):

    # The parameters
    a, b, n = .7725, .601, .5189
    age = 4.56
    mass, feh, d, Av = 1., 0., 10., 0.
    params = np.array([a, b, n, np.log(age), np.log(mass), feh, np.log(d),
                       Av])

    J, J_err = 3.711, .01  # absolute magnitudes/apparent at D = 10pc
    H, H_err = 3.453, .01
    K, K_err = 3.357, .01

    bands = dict(J=(J, J_err), H=(H, H_err), K=(K, K_err),)
    parallax, period, bv = (1./d*1e3, .01), (26.6098830128, 1.), (.65, .01)
    Teff, logg, feh = (5778, 50), (4.43812, .01), (0., .01)

    # test the iso_lnlike
    mist = MIST_Isochrone()

    # iso_lnlike preamble.
    if spec:
        mod = StarModel(mist, Teff=Teff, logg=logg, feh=feh,
                        parallax=parallax, use_emcee=True)
    else:
        mod = StarModel(mist, J=(J, J_err), H=(H, H_err), K=(K, K_err))

    # Run emcee
    nwalkers, nsteps, ndim = 64, 10000, len(params)
    p0 = [1e-4*np.random.rand(ndim) + params for i in range(nwalkers)]

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                    args=[mod, period, bv])
    print("burning in...")
    pos, _, _ = sampler.run_mcmc(p0, 1000)
    sampler.reset()
    print("production run...")
    sampler.run_mcmc(pos, nsteps)
    flat = np.reshape(sampler.chain, (nwalkers*nsteps, ndim))
    result = [np.median(flat[:, i]) for i in range(len(flat[0, :]))]
    print(result)

    if plot:
        truths = [.7725, .601, .5189, np.log(4.56), np.log(1), 0., np.log(d),
                  0.]
        print(truths)
        labels = ["$a$", "$b$", "$n$", "$\ln(Age)$", "$\ln(Mass)$",
                  "$[Fe/H]$", "$\ln(D)$", "$A_v$"]
        fig = corner.corner(flat, labels=labels, truths=truths)
        fig.savefig(os.path.join(FIG_DIR, "corner_single"))

        # Plot probability trace.
        plt.clf()
        plt.plot(sampler.lnprobability.T, "k")
        plt.savefig(os.path.join(FIG_DIR, "prob_trace_single"))

        # Plot individual parameter traces.
        for i in range(ndim):
            plt.clf()
            plt.plot(sampler.chain[:, :,  i].T, alpha=.5)
            plt.ylabel(labels[i])
            plt.savefig(os.path.join(FIG_DIR, "{}_trace_single".format(i)))

    f = h5py.File(os.path.join(RESULTS_DIR, "{}.h5".format(fname)), "w")
    data = f.create_dataset("samples", np.shape(flat))
    data[:, :] = flat
    f.close()


def age_hist(fname):
    """
    Make a simple 1d histogram of the marginal posterior for age using
    different methods.
    params:
    -------
    fname: (str)
        The name of the h5 results file *and* the name of the figure.
    """

    with h5py.File(os.path.join(RESULTS_DIR, "all_single2.h5"),
                   "r") as f:
        flat0 = f["samples"][...]
    with h5py.File(os.path.join(RESULTS_DIR,
                                "all_spec_single2.h5".format(fname)),
                   "r") as f:
        flat1 = f["samples"][...]
    with h5py.File(os.path.join(RESULTS_DIR, "gyro_single2.h5".format(fname)),
                   "r") as f:
        flat2 = f["samples"][...]

    plt.clf()
    n = 15
    plt.hist(np.exp(flat0[:, 3]), n, normed=True, color=".7", alpha=.5,
             label="$\mathrm{Colours}$")
    plt.hist(np.exp(flat1[:, 3]), n, normed=True, color=".3", alpha=.5,
             label="$\mathrm{Spectra}$")
    plt.hist(np.exp(flat2[:, 6]), n, normed=True, color="k", alpha=.8,
             label="$\mathrm{Gyro}$")
    plt.legend()
    plt.xlabel("$\mathrm{Age~(Gyr)}$")
    plt.axvline(4.567)
    plt.subplots_adjust(bottom=.15)
    plt.savefig(os.path.join(FIG_DIR, fname))

0
if __name__ == "__main__":
    FIG_DIR = "/Users/ruthangus/projects/chronometer/chronometer/figures"
    RESULTS_DIR = "/Users/ruthangus/projects/chronometer/chronometer/results"
    # sun_demo("all_single", gyro=False, iso=True, plot=False, spec=False)
    # sun_demo("all_spec_single", gyro=False, iso=True, plot=False, spec=True)
    # from models import gc_model
    # par = [.7725, .601, .5189]
    # print(gc_model(par, np.log(4.567), .65))
    # assert 0
    # sun_demo("gyro_single", gyro=True, iso=True, plot=False, spec=True)
    age_hist("all_hist")
