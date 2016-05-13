# -*- coding: utf-8 -*-
"""
Code written by Chi Nguyen.
"""

from __future__ import division, print_function
from numpy.polynomial import polynomial as P
from os import path
import matplotlib

import emcee
import corner
import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as pl
from matplotlib.ticker import MaxNLocator
matplotlib.use('Agg')

# group modulus
import density as ds

# Reproducible results!
np.random.seed(123)

def chis_code(m_1, m_2, s, rho, q, q_err, eta, M_c, M_c_err, V,
              output_directory):
    # Transform M_c into log-space.
    T = 1
    N = len(M_c)
    x = np.log10(M_c)
    y = ds.formation_rate_estimator(M_c,V,T,bandwidth="scott")
    #y = np.log10(y)
    yerr = 0.1*np.random.normal(0,1,len(x))
    degree = 13 # degree of polynomial
    # least square fitting
    lam_ls,cov_ls = least_square(x,y,yerr,degree,output_directory)
    lam_ml = maximum_likelihood(x,y,yerr,degree,lam_ls,output_directory)
    lam_mcmc = MCMC(x,y,yerr,degree,lam_ls,output_directory)
    
    return lam_mcmc

# Define a polynomial
def poly_model(x, degree):
    Y = np.column_stack(x**i for i in range(degree+1))
    return Y
    
# Define the probability function as likelihood * prior.

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

def lnprior(theta):
    """
    Example usage of polynomial smoothing prior.
    """
    # Polynomial is 0 + 1x + ... + 4x^4.
    p = np.arange(2)
    # Smooth derivatives of 3rd degree and higher.
    smoothing_degree = 1
    # Integrate from 0 to 2.
    xmin, xmax = 0, 2
    # Do not scale result.
    gamma = 1

    # Run the function and display the result.
    lnp = smoothing_poly_lnprior(p, smoothing_degree, xmin, xmax, gamma)

    return lnp

def lnlike(theta, x, y, yerr):
    lam_ml = list(reversed(theta))
    model = np.polyval(lam_ml,x)
    inv_sigma2 = 1.0/(yerr**2) #+ model**2*np.exp(2*lnf))
    return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))

def lnprob(theta, x, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)

def chi2(*args):
    return -2 * lnlike(*args)

def least_square(x,y,yerr,degree,output_directory):
    # Plot the dataset and the true model.
    fig,ax = pl.subplots()
    ax.errorbar(x, y, yerr=yerr, fmt=".k")
    
    # Do the least-squares fit and compute the uncertainties.
    A = poly_model(x, degree)
    C = np.diag(yerr * yerr)
    cov = np.linalg.inv(np.dot(A.T, np.linalg.solve(C, A)))
    lam_ls = np.array([0]*degree)
    lam_ls = np.dot(cov, np.dot(A.T, np.linalg.solve(C, y)))
    print("Least square fit coefficients:\n",lam_ls)
    print("Least square fit uncertainty:\n",np.diag(cov))
    
    # Plot the least-square result.
    xl = np.linspace(-0.1, 2,1000)
    lam_ls_for_plot = list(reversed(lam_ls)) # reverse the order of the coefficient, 
                                      # so that lam_to_plot[0] = the coefficient of the highest power
    ax.plot(xl,np.polyval(lam_ls_for_plot,xl), "-r")
    ax.set_ylim(-4,4)
    pl.savefig(path.join(output_directory, "line-least-squares.pdf"))
    pl.show()
    
    return (lam_ls,cov)

def maximum_likelihood(x,y,yerr,degree,lam_ls,output_directory):
    # Find the maximum likelihood value.
    result = op.minimize(chi2, [lam_ls], args=(x, y, yerr))
    lam_ml = result["x"]
    print("Maximum likelihood fit coefficients:\n",lam_ml)

    # Plot the maximum likelihood result.
    xl = np.linspace(-0.1, 2,1000)
    lam_ml_for_plot = list(reversed(lam_ml))
    fig,ax = pl.subplots()
    ax.errorbar(x, y, yerr=yerr, fmt=".k")
    ax.plot(xl,np.polyval(lam_ml_for_plot,xl), "c", lw=2)
    ax.set_ylim(-4,4)
    pl.savefig(path.join(output_directory, "line-max-likelihood.pdf"))
    pl.show()

    return lam_ml


def MCMC(x,y,yerr,degree,lam_ml,output_directory):                                   
    # Set up the sampler.
    ndim, nwalkers = degree+1, 30
    pos = [lam_ml + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, yerr))

    # Clear and run the production chain.
    print("Running MCMC...")
    sampler.run_mcmc(pos, 2000, rstate0=np.random.get_state())
    print("Done.")
    

    pl.clf()
    fig, axes = pl.subplots(3, 1, sharex=True, figsize=(8, 9))
    axes[0].plot(sampler.chain[:, :, 10].T, color="k", alpha=0.4)
    axes[0].set_ylabel("$\lambda_0$")

    axes[1].plot(sampler.chain[:, :, 11].T, color="k", alpha=0.4)
    axes[1].set_ylabel("$\lambda_1$")

    axes[2].plot(np.exp(sampler.chain[:, :, 12]).T, color="k", alpha=0.4)
    axes[2].set_ylabel("$\lambda_2$")
    axes[2].set_xlabel("step number")


    fig.tight_layout(h_pad=0.0)
    fig.savefig(path.join(output_directory, "line-time.png"))
    pl.show()

    # Make the triangle plot.
    burnin = 50
    samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))

    print("MCMC fit coefficient:\n")
    print(samples[0])
    
    lam_MCMC = samples[len(samples)-1]
    fig = corner.corner(samples)
    fig.savefig(path.join(output_directory, "line-triangle.jpg"))

    pl.figure()
    
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
    #ax.plot(xl, y_low, 'k--')
    ax2.set_ylim(-4,4)
    pl.savefig(path.join(output_directory, "line-MCMC.pdf"))
    pl.show()


    # Plot some samples onto the data.
    #for m, b, lnf in samples[np.random.randint(len(samples), size=100)]:
    #pl.plot(xl, m*xl+b, color="k", alpha=0.1)
    #pl.plot(x, y, color="r", lw=2, alpha=0.8)
    #pl.errorbar(x, y, yerr=yerr, fmt=".k")
    #pl.ylim(-9, 9)
    #pl.xlabel("$x$")
    #pl.ylabel("$y$")
    #pl.tight_layout()
    #pl.show()
    #pl.savefig("line-mcmc.png")

    # Compute the quantiles.
   # samples[:, 2] = np.exp(samples[:, 2])
    #m_mcmc, b_mcmc, f_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
     #                        zip(*np.percentile(samples, [16, 50, 84],
                                   #             axis=0)))
   # print("""MCMC result:
    #    m = {0[0]} +{0[1]} -{0[2]} 
     #   b = {1[0]} +{1[1]} -{1[2]} 
      #  f = {2[0]} +{2[1]} -{2[2]}
    #""".format(m_mcmc, b_mcmc, f_mcmc))
    return lam_MCMC
