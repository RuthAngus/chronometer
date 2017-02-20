import numpy as np
from action_age_evolution import calc_dispersion
import emcee
import corner
import matplotlib.pyplot as plt

plotpar = {'axes.labelsize': 18,
           'font.size': 10,
           'legend.fontsize': 15,
           'xtick.labelsize': 18,
           'ytick.labelsize': 18,
           'text.usetex': True}
plt.rcParams.update(plotpar)

def lnprob(pars, x, y, yerr):
    sz0, t1, beta, R0, Rc, hsz = pars
    model = calc_dispersion([np.exp(sz0), np.exp(t1), beta, np.exp(R0),
                                    np.exp(Rc), np.exp(hsz)], x)
    return sum(-.5*((model - y)/yerr)**2) + lnprior(pars)


def lnprior(pars):
    lnsz0, lnt1, beta, lnR0, lnRc, lnhsz = pars
    if -20 < lnsz0 < 20 and -20 < lnt1 < 20 and -100 < beta < 100 \
            and -20 < lnR0 < 20 and -20 < lnRc < 20 and -20 < lnhsz < 20:
        return 0.
    else:
        return -np.inf


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
    solar_radius = 8.
    hr = 2.68/solar_radius

    # Today
    sr = 34.
    sz = 25.1

    zpar_init = np.array([np.log(sz0), np.log(t1), beta, np.log(R0),
                          np.log(Rc), np.log(hsz)])
    rpar_init = np.array([np.log(sr0), np.log(t1), beta, np.log(R0),
                          np.log(Rc), np.log(hsz)])

    sigma_z = calc_dispersion([sz0 + 5, t1, beta + .2, R0, Rc, hsz], time)
    sigma_r = calc_dispersion([sr0 + 5, t1, beta + .2, R0, Rc, hsz], time)

    print(lnprob(zpar_init, time, sigma_z, sigma_z*.1))
    x, y, yerr = time, sigma_z, sigma_z*.1

    ndim, nwalkers, nsteps = len(zpar_init), 24, 10000
    p0 = [1e-4*np.random.rand(ndim) + zpar_init for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[x, y, yerr])
    pos, _, _ = sampler.run_mcmc(p0, 500)
    sampler.reset()
    sampler.run_mcmc(pos, nsteps)
    flat = np.reshape(sampler.chain, (nwalkers*nsteps, ndim))

    # flat[:, :2] = np.exp(flat[:, :2])
    # flat[:, 3:] = np.exp(flat[:, 3:])
    labels = ["$\ln \sigma_{z0}$", "$t_1$", "$\\beta$", "$R_0$", "$R_c$",
              "$\sigma_{Hz}$"]
    fig = corner.corner(flat, labels=labels)
    fig.savefig("zcorner")
