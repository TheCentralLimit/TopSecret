"""
Markov Chain Monte Carlo functions.
"""
from __future__ import division, print_function
from numpy.polynomial import polynomial as P
from os import path
#import matplotlib

import emcee
import corner
import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as pl
from matplotlib.ticker import MaxNLocator
import math
#matplotlib.use('Agg')

# group modulus
import density as ds
import mcmc

# Reproducible results!
np.random.seed(123)

# Define a polynomial model
def poly_model(x, degree):
    Y = np.column_stack(x**i for i in range(degree+1))
    return Y

def smoothing_poly_lnprior(poly, order, xmin, xmax, gamma=1):
    """
    A smoothing prior that suppresses higher order derivatives of a polynomial,
    poly = a + b x + c x*x + ..., described by a vector of its coefficients,
    [a, b, c, ...].

    Functional form is:

    ln p(poly coeffs) =
      -gamma * integrate( (diff(poly(x), x, order))^2, x, xmin, xmax)

    So it takes the `order`th derivative of the polynomial, squares it,
    integrates that from xmin to xmax, and scales by -gamma.
    """
    # Take the `degree`th derivative of the polynomial.
    poly_diff = P.polyder(poly, m=order)
    # Square the polynomial.
    poly_diff_sq = P.polypow(poly_diff, 2)
    # Take the indefinite integral of the polynomial.
    poly_int_indef = P.polyint(poly_diff_sq)
    # Evaluate the integral at xmin and xmax to get the definite integral.
    poly_int_def = (
        P.polyval(xmax, poly_int_indef) - P.polyval(xmin, poly_int_indef)
    )
    # Scale by -gamma to get the log prior
    lnp = -gamma * poly_int_def

    return lnp


def lnlike(theta, x, y, yerr):
    model = np.polyval(theta, x)

    return -0.5 * np.linalg.norm((y-model) * yerr**-2)**2


def lnprob(theta, order, x, y, yerr, xmin, xmax, gamma):
    lp = smoothing_poly_lnprior(theta, order, xmin, xmax, gamma)
    ll = lnlike(theta, x, y, yerr)

    if not (np.isfinite(lp) and np.isfinite(ll)):
        return -np.inf

    return lp + ll


def chi2(*args):
    return -2 * lnlike(*args)


# define least-square to provide initial guess for MCMC
def least_square(x, y, yerr, degree):
    # Do the least-squares fit and compute the uncertainties.
    A = poly_model(x, degree)
    lam_ls,residual,rank,s = np.linalg.lstsq(A,y)
    print("Least square fit coefficients:\n",lam_ls)
    return lam_ls


def maximum_likelihood(x, y, yerr, degree, lam_ls, output_directory):
    # Find the maximum likelihood value.
    result = op.minimize(chi2, [lam_ls], args=(x, y, yerr))
    lam_ml = result["x"]
    print("Maximum likelihood fit coefficients:\n",lam_ml)
    return lam_ml

def MCMC(x, y, yerr, degree, order, ax_density, output_directory):
    # Prior conditions
    xmin, xmax = 1e-1, 1e+2
    gamma = 1

    pred_lstsq = least_square(x, y, yerr, degree)

    # Set up the sampler.
    ndim, nwalkers = degree+1, 100
    pos = [pred_lstsq + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
    args = (order, x, y, yerr, xmin, xmax, gamma)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=args)

    # Clear and run the production chain.
    print("Running MCMC...")
    sampler.run_mcmc(pos, 1000, rstate0=np.random.get_state())
    print("Done.")

    fig, axes = pl.subplots(5, 1, sharex=True, figsize=(8, 9))
    axes[0].plot(sampler.chain[:, :, 0].T, color="k", alpha=0.4)
    axes[0].set_ylabel("$\lambda_0$")

    axes[1].plot(sampler.chain[:, :, 1].T, color="k", alpha=0.4)
    axes[1].set_ylabel("$\lambda_1$")

    axes[2].plot(np.exp(sampler.chain[:, :, 2]).T, color="k", alpha=0.4)
    axes[2].set_ylabel("$\lambda_2$")
    axes[2].set_xlabel("step number")

    axes[3].plot(np.exp(sampler.chain[:, :, 3]).T, color="k", alpha=0.4)
    axes[3].set_ylabel("$\lambda_2$")
    axes[3].set_xlabel("step number")

    axes[4].plot(np.exp(sampler.chain[:, :, 4]).T, color="k", alpha=0.4)
    axes[4].set_ylabel("$\lambda_2$")
    axes[4].set_xlabel("step number")


    fig.tight_layout(h_pad=0.0)
    fig.savefig(path.join(output_directory, "line-time.pdf"))

    # Make the triangle plot.
    burnin = 500
    samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))

    xl = np.logspace(np.log10(x.min()), np.log10(x.max()/10), 1000)

    S, _ = np.shape(samples)
    pred_1 = np.zeros_like(xl)
    pred_2 = np.zeros_like(xl)

    for sample in samples:
        pred_i = np.polyval(sample, xl)

        pred_1 += pred_i / S
        pred_2 += pred_i**2 / S

    pred_MCMC = pred_1
    err_MCMC = np.sqrt((pred_2 - pred_1**2) / S)

    ax_density.plot(xl, pred_MCMC, "g-")
    ax_density.fill_between(xl, pred_MCMC-err_MCMC, pred_MCMC+err_MCMC,
                            color="green", alpha=0.2)

    return

    # Add a plot of the result. (note the fit to f is a fit to error)
    fig,ax2 = pl.subplots()




    lam_MCMC_for_plot = lam_MCMC
    yvals = np.polyval(lam_MCMC_for_plot,xl)
    y_low = np.percentile(yvals,5)
    y_high = np.percentile(yvals,95)
    ax2.errorbar(x, y, yerr=yerr, fmt=".k")
    ax2.plot(xl, yvals, 'c')
    #ax2.plot(xl, y_low, 'k--')
    #ax2.plot(xl, y_high, 'k--')
    #ax2.set_ylim(-4,4)
    pl.savefig(path.join(output_directory, "line-MCMC.pdf"))


    # Plot some samples onto the data.
    for lam in samples[np.random.randint(len(samples), size=10)]:
        pl.plot(xl,np.polyval(lam,xl), color="k", alpha=0.1)
    #pl.plot(x, y, color="r", lw=2, alpha=0.8)
    pl.errorbar(x, y, yerr=yerr, fmt=".k")

    #pl.tight_layout()
    pl.savefig(path.join(output_directory, "line-mcmc.pdf"))

    # Compute the quantiles.
    samples[:, 2] = np.exp(samples[:, 2])
    lam_MCMC_best = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],
                                                axis=0)))
    print("MCMC fit coefficient:\n")
    print(lam_MCMC_best[1][:])

    print("MCMC uncertainty:\n")
    print(lam_MCMC_best[2][:])

    # Add a plot of the result. (note the fit to f is a fit to error)
    # fig,ax2 = pl.subplots()
    # xl = np.linspace(-0.1, 2,1000)
    # lam_MCMC_for_plot = lam_MCMC_best[1][:]
    # lam_MCMC_uncertainty = lam_MCMC_best[2][:]
    # y_high = np.zeros(len(xl))
    # y_low = np.zeros(len(xl))
    # yvals = np.polyval(lam_MCMC_for_plot,xl)
    # y_low = np.polyval(lam_MCMC_for_plot-lam_MCMC_uncertainty,xl)
    # y_high = np.polyval(lam_MCMC_for_plot+lam_MCMC_uncertainty,xl)
    # ax2.errorbar(x, y, yerr=yerr, fmt=".k")
    # ax2.plot(xl, yvals, 'c')
    # ax2.plot(xl, y_low, 'k--')
    # ax2.plot(xl, y_high, 'k--')
    #ax2.set_ylim(-4,4)
    pl.savefig(path.join(output_directory, "line-MCMC.pdf"))

    fig = corner.corner(samples)
    fig.savefig(path.join(output_directory, "line-triangle.pdf"))

    return lam_MCMC_best
