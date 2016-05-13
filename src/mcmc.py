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

def smoothing_poly_lnprior(poly, degree, xmin, xmax, gamma=1):
    """
    A smoothing prior that suppresses higher order derivatives of a polynomial,
    poly = a + b x + c x*x + ..., described by a vector of its coefficients,
    [a, b, c, ...].

    Functional form is:

    ln p(poly coeffs) =
      -gamma * integrate( (diff(poly(x), x, degree))^2, x, xmin, xmax)

    So it takes the `degree`th derivative of the polynomial, squares it,
    integrates that from xmin to xmax, and scales by -gamma.
    """
    # Take the `degree`th derivative of the polynomial.
    poly_diff = P.polyder(poly, m=degree)
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

def lnprior(theta,degree):
    """
    Example usage of polynomial smoothing prior.
    """
    # Polynomial is 0 + 1x + ... + 4x^4.
    p = np.arange(2)
    # Smooth derivatives of 3rd degree and higher.
    smoothing_degree = 3
    # Integrate from 0 to 2.
    xmin, xmax = 0, 2
    # Do not scale result.
    gamma = 1

    # Run the function and display the result.
    lnp = np.log(smoothing_poly_lnprior(p, smoothing_degree, xmin, xmax, gamma))

    return lnp 

def lnlike(theta, x, y, yerr):
    lam_ml = list(reversed(theta))
    model = np.polyval(lam_ml,x)
    inv_sigma2 = 1.0/(yerr**2) #+ model**2*np.exp(2*lnf))
    return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))

def lnprob(theta, degree, x, y, yerr):
    lp = lnprior(theta,degree)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)

def chi2(*args):
    return -2 * lnlike(*args)

# define least-square to provide initial guess for MCMC
def least_square(x,y,yerr,degree,output_directory):
    # Plot the dataset and the true model.
    fig,ax = pl.subplots()
    ax.errorbar(x, y, yerr=yerr, fmt=".k")
    
    # Do the least-squares fit and compute the uncertainties.
    A = poly_model(x, degree)
    lam_ls,residual,rank,s = np.linalg.lstsq(A,y)
    print("Least square fit coefficients:\n",lam_ls)
    return lam_ls

def maximum_likelihood(x,y,yerr,degree,lam_ls,output_directory):
    # Find the maximum likelihood value.
    result = op.minimize(chi2, [lam_ls], args=(x, y, yerr))
    lam_ml = result["x"]
    print("Maximum likelihood fit coefficients:\n",lam_ml)
    return lam_ml

def MCMC(x,y,yerr,degree,lam_ml,output_directory):                                   
    # Set up the sampler.
    ndim, nwalkers = degree+1, 100
    pos = [lam_ml + 1e-15*np.random.randn(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(degree, x, y, yerr))

    # Clear and run the production chain.
    print("Running MCMC...")
    sampler.run_mcmc(pos, 1000, rstate0=np.random.get_state())
    print("Done.")
    

    pl.clf()
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
    fig.savefig(path.join(output_directory, "line-time.png"))
    pl.show()

    # Make the triangle plot.
    burnin = 50
    samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
    
    lam_MCMC = samples[len(samples)-1]
    
    # Add a plot of the result. (note the fit to f is a fit to error)
    fig,ax2 = pl.subplots()
    xl = np.linspace(-0.1, 2,1000)
    lam_MCMC_for_plot = list(reversed(lam_MCMC))
    y_high = np.zeros(len(xl))
    y_low = np.zeros(len(xl))
    yvals = np.polyval(lam_MCMC_for_plot,xl)
    y_low = np.percentile(yvals,5)
    y_high = np.percentile(yvals,95)
    ax2.errorbar(x, y, yerr=yerr, fmt=".k")
    ax2.plot(xl, yvals, 'c')
    #ax2.plot(xl, y_low, 'k--')
    #ax2.plot(xl, y_high, 'k--')
    #ax2.set_ylim(-4,4)
    pl.savefig(path.join(output_directory, "line-MCMC.pdf"))
    pl.show()


    # Plot some samples onto the data.
    for lam in samples[np.random.randint(len(samples), size=10)]:
        lam_for_plot = list(reversed(lam))
        pl.plot(xl,np.polyval(lam_for_plot,xl), color="k", alpha=0.1)
    #pl.plot(x, y, color="r", lw=2, alpha=0.8)
    pl.errorbar(x, y, yerr=yerr, fmt=".k")
       
    #pl.tight_layout()
    pl.show()
    pl.savefig(path.join(output_directory, "line-mcmc.png"))

    # Compute the quantiles.
    samples[:, 2] = np.exp(samples[:, 2])
    lam_MCMC_best = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],
                                                axis=0)))
    print("MCMC fit coefficient:\n")
    print(lam_MCMC_best[1][:])
    
    print("MCMC uncertainty:\n")
    print(lam_MCMC_best[2][:])

    
    fig = corner.corner(samples)
    fig.savefig(path.join(output_directory, "line-triangle.jpg"))

    pl.figure()
    
    
    return lam_MCMC_best

