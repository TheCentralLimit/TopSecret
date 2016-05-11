# -*- coding: utf-8 -*-
"""
Code written by Chi Nguyen.
"""

from __future__ import division, print_function
from os import path
#import matplotlib
#matplotlib.use('Agg')
import emcee
import corner
import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as pl
from matplotlib.ticker import MaxNLocator

# group modulus
import density as ds

# Reproducible results!
np.random.seed(123)

def chis_code(m_1, m_2, s, rho, q, q_err, eta, M_c, M_c_err, V,
              output_directory):
    fit = fitting_MCMC(M_c,M_c_err,V,1)
    return (M_c,V,1)

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
    smoothing_degree = 3
    # Integrate from 0 to 2.
    xmin, xmax = 0, 2
    # Do not scale result.
    gamma = 1

    # Run the function and display the result.
    lnp = smoothing_poly_lnprior(p, smoothing_degree, xmin, xmax, gamma)

    return lnp

def lnlike(theta, x, y, yerr):
    m, b, lnf = theta
    model = m * x + b
    inv_sigma2 = 1.0/(yerr**2 + model**2*np.exp(2*lnf))
    return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))

def lnprob(theta, x, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)


def fitting_MCMC(M_c,M_c_err,V,T):
    # Transform M_c into log-space.
    N = len(M_c)
    x = np.log10(M_c)
    y = ds.formation_rate_estimator(M_c,V,T,bandwidth="scott")
    y = np.log10(y)
    yerr = np.log10(M_c_err)
    lnf = 0.05
    degree = 5
    # Plot the dataset and the true model.
    fig,ax = pl.subplots()
    ax.errorbar(x, y, yerr=yerr, fmt=".k")

    # Do the least-squares fit and compute the uncertainties.
    A = poly_model(x, degree)
    C = np.diag(yerr * yerr)
    cov = np.linalg.inv(np.dot(A.T, np.linalg.solve(C, A)))
    lam = np.array([0]*degree)
    lam = np.dot(cov, np.dot(A.T, np.linalg.solve(C, y)))
    print('Fit coefficients:\n',lam)
    print('Fit uncertainty:\n',np.diag(cov))

    # Plot the least-squares result.
    xl = np.linspace(-0.1, 2,1000)
    lam_to_plot = list(reversed(lam)) # reverse the order of the coefficient, 
                                      # so that lam_to_plot[0] = the coefficient of the highest power
    ax.plot(xl,np.polyval(lam_to_plot,xl), "-r")
    ax.set_ylim(-4,1.5)
    pl.savefig("line-least-squares.pdf")
    pl.show()

    # Find the maximum likelihood value.
    chi2 = lambda *args: -2 * lnlike(*args)
    result = op.minimize(chi2, [m_ls, b_ls,lnf], args=(x, y, yerr))
    m_ml, b_ml, lnf_ml = result["x"]
    #print("""Maximum likelihood result:
     #   m = {0} 
      #  b = {1}
       # f = {2} 
    #""".format(m_ml, b_ml, np.exp(lnf_ml)))

    # Plot the maximum likelihood result.
    #pl.plot(xl, m_ml*xl+b_ml, "k", lw=2)
    #pl.savefig("line-max-likelihood.pdf")
    #pl.show()

    # Set up the sampler.
    ndim, nwalkers = 3, 100
    pos = [result["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, yerr))

    # Clear and run the production chain.
    print("Running MCMC...")
    sampler.run_mcmc(pos, 1000, rstate0=np.random.get_state())
    print("Done.")

    pl.clf()
    fig, axes = pl.subplots(3, 1, sharex=True, figsize=(8, 9))
    axes[0].plot(sampler.chain[:, :, 0].T, color="k", alpha=0.4)
    axes[0].yaxis.set_major_locator(MaxNLocator(5))
    #axes[0].axhline(m_true, color="#888888", lw=2)
    axes[0].set_ylabel("$m$")

    axes[1].plot(sampler.chain[:, :, 1].T, color="k", alpha=0.4)
    axes[1].yaxis.set_major_locator(MaxNLocator(5))
    #axes[1].axhline(b_true, color="#888888", lw=2)
    axes[1].set_ylabel("$b$")

    axes[2].plot(np.exp(sampler.chain[:, :, 2]).T, color="k", alpha=0.4)
    axes[2].yaxis.set_major_locator(MaxNLocator(5))
    #axes[2].axhline(f_true, color="#888888", lw=2)
    axes[2].set_ylabel("$f$")
    axes[2].set_xlabel("step number")

    fig.tight_layout(h_pad=0.0)
    #fig.savefig("line-time.png")
    pl.show()

    # Make the triangle plot.
    burnin = 50
    samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))

    #fig = corner.corner(samples, labels=["$m$", "$b$", "$\ln\,f$"],
                     # truths=[m_true, b_true, np.log(f_true)])
    #fig.savefig("line-triangle.jpg")

    pl.figure()
    # Add a plot of the result. (note the fit to f is a fit to error)
    xvals = np.linspace(0,10,30)
    y_high = np.zeros(len(xvals))
    y_low = np.zeros(len(xvals))
    for indx in np.arange(len(xvals)):
        yvals = samples[:,0]*xvals[indx] + samples[:,1]
        y_low[indx] = np.percentile(yvals,5)
        y_high[indx] = np.percentile(yvals,95)
    pl.plot(xvals, y_high, 'k--')
    pl.plot(xvals, y_low, 'k--')


    # Plot some samples onto the data.
    #for m, b, lnf in samples[np.random.randint(len(samples), size=100)]:
    #pl.plot(xl, m*xl+b, color="k", alpha=0.1)
    #pl.plot(x, y, color="r", lw=2, alpha=0.8)
    pl.errorbar(x, y, yerr=yerr, fmt=".k")
    pl.ylim(-9, 9)
    pl.xlabel("$x$")
    pl.ylabel("$y$")
    pl.tight_layout()
    pl.show()
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
    return 0
