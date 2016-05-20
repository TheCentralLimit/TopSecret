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

def lnprior(theta,degree,xmin,xmax):
    xmin = xmin - 1.0
    xmax = xmax + 1.0
    """
    Example usage of polynomial smoothing prior.
    """
    # Polynomial is 0 + 1x + ... + 4x^4.
    p = theta
    # Smooth derivatives of 3rd degree and higher.
    smoothing_degree = 3
    # Integrate from xmin to xmax
    # Do not scale result.
    gamma = 0.5

    # Run the function and display the result.
    lnp = smoothing_poly_lnprior(p, smoothing_degree, xmin, xmax, gamma)

    return lnp 

def lnlike(theta, x, y, yerr):
 
    model = np.polyval(theta,x)
    inv_sigma2 = 1.0/(yerr**2) #+ model**2*np.exp(2*lnf))
    return -0.5*(np.sum((y-model)**2*inv_sigma2))# - np.log(inv_sigma2)))

def lnprob(theta, degree, x, y, yerr):
    xmin = np.amin(x)
    xmax = np.amax(x)
    lp = lnprior(theta,degree,xmin,xmax)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)

def chi2(*args):
    return -2 * lnlike(*args)

# define least-square to provide initial guess for MCMC
def least_square(x,y,yerr,degree,output_directory):
    # Do the least-squares fit
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
    ndim, nwalkers = degree+1, 50
    pos = [lam_ml + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(degree, x, y, yerr))

    # Clear and run the production chain.
    print("Running MCMC...")
    sampler.run_mcmc(pos, 5000, rstate0=np.random.get_state())
    print("Done.")
    

    #pl.clf()
    fig, axes = pl.subplots(5, 1, sharex=True, figsize=(8, 9))
    axes[0].plot(sampler.chain[:, :, 0].T, color="k", alpha=0.4)
    axes[0].set_ylabel("$\lambda_0$")

    axes[1].plot(sampler.chain[:, :, 1].T, color="k", alpha=0.4)
    axes[1].set_ylabel("$\lambda_1$")

    axes[2].plot(np.exp(sampler.chain[:, :, 2]).T, color="k", alpha=0.4)
    axes[2].set_ylabel("$\lambda_2$")
    axes[2].set_xlabel("step number")
    
    axes[3].plot(np.exp(sampler.chain[:, :, 3]).T, color="k", alpha=0.4)
    axes[3].set_ylabel("$\lambda_3$")
    axes[3].set_xlabel("step number")
    
    axes[4].plot(np.exp(sampler.chain[:, :, 4]).T, color="k", alpha=0.4)
    axes[4].set_ylabel("$\lambda_4$")
    axes[4].set_xlabel("step number")
    
    fig.tight_layout(h_pad=0.0)
    fig.savefig(path.join(output_directory, "line-time-1.pdf"))
    
    pl.show()
    
    fig, axes = pl.subplots(5, 1, sharex=True, figsize=(8, 9))
    axes[0].plot(np.exp(sampler.chain[:, :, 5]).T, color="k", alpha=0.4)
    axes[0].set_ylabel("$\lambda_5$")
    axes[0].set_xlabel("step number")
    
    axes[1].plot(np.exp(sampler.chain[:, :, 6]).T, color="k", alpha=0.4)
    axes[1].set_ylabel("$\lambda_6$")
    axes[1].set_xlabel("step number")
   
    axes[2].plot(np.exp(sampler.chain[:, :, 7]).T, color="k", alpha=0.4)
    axes[2].set_ylabel("$\lambda_7$")
    axes[2].set_xlabel("step number")
    
    axes[3].plot(np.exp(sampler.chain[:, :, 8]).T, color="k", alpha=0.4)
    axes[3].set_ylabel("$\lambda_8$")
    axes[3].set_xlabel("step number")
    
    axes[4].plot(np.exp(sampler.chain[:, :, 9]).T, color="k", alpha=0.4)
    axes[4].set_ylabel("$\lambda_9$")
    axes[4].set_xlabel("step number")

    fig.tight_layout(h_pad=0.0)
    fig.savefig(path.join(output_directory, "line-time-2.pdf"))
    
    pl.show()

    # Make the triangle plot.
    burnin = 3500
    samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
    
    lam_MCMC = samples[len(samples)-1]
    
    # Add a plot of the result. (note the fit to f is a fit to error)
    fig,ax2 = pl.subplots()
    xl = np.linspace(-0.1, 1.9,1000)
    y_high = np.zeros(len(xl))
    y_low = np.zeros(len(xl))
    yvals = np.polyval(lam_MCMC,xl)
    y_low = np.percentile(yvals,5)
    y_high = np.percentile(yvals,95)
    ax2.errorbar(x, y, yerr=yerr, fmt=".k")
    ax2.plot(xl, yvals, "c")
    ax2.set_xlabel("$\log(\mathcal{M}_c)$")
    ax2.set_ylabel("$r(\mathcal{M}_c)$")
    ax2.set_yscale("log")
    ax2.set_ylim(10**(-10), 10**(-5))
    pl.savefig(path.join(output_directory, "line-MCMC.pdf"))
    pl.show()


    # Plot some samples onto the data.
    fig,ax3 = pl.subplots()
    for lam in samples[np.random.randint(len(samples), size=200)]:
        ax3.plot(xl,np.polyval(lam,xl), color="k", alpha=0.1)
    ax3.plot(x, y, ".r", alpha=0.8)
    ax3.set_yscale("log")
    ax3.set_xlabel("$\log(\mathcal{M}_c)$")
    ax3.set_ylabel("$r(\mathcal{M}_c)$")
    ax3.set_ylim(10**(-10.),10**(-5.))
    ax3.errorbar(x, y, yerr=yerr, fmt=".k")   
    #pl.tight_layout()
    pl.savefig(path.join(output_directory, "line-mcmc_err.pdf"))
    pl.show()
    
    # Compute the quantiles.
    samples[:, 2] = np.exp(samples[:, 2])
    lam_MCMC_best = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],
                                                axis=0)))
    fig = corner.corner(samples)
    fig.savefig(path.join(output_directory, "line-triangle.pdf"))
    
    pl.figure()
    
    print("MCMC best fit:\n")
    print(lam_MCMC_best)

    
    return lam_MCMC_best

