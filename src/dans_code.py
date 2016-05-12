"""
Code written by Daniel Wysocki.
"""

from __future__ import division, print_function

from astroML.density_estimation import knuth_bin_width
import numpy as np
from numpy.polynomial import polynomial as P
import matplotlib as mpl
import matplotlib.pyplot as plt
from statsmodels.nonparametric.kde import KDEUnivariate

from os import path

from utils import pairwise


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


def smoothing_poly_lnprior_example():
    """
    Example usage of polynomial smoothing prior.
    """
    # Polynomial is 0 + 1x + ... + 4x^4.
    p = np.arange(5)
    # Smooth derivatives of 3rd degree and higher.
    smoothing_degree = 3
    # Integrate from 0 to 2.
    xmin, xmax = 0, 2
    # Do not scale result.
    gamma = 1

    # Run the function and display the result.
    lnp = smoothing_poly_lnprior(p, smoothing_degree, xmin, xmax, gamma)
    print("Smoothing prior example")
    print("lnp =", lnp)


def knuth_n_bins(data):
    bandwidth = knuth_bin_width(data)
    return np.ceil((np.max(data) - np.min(data)) / bandwidth)



def hist_pdf(x, bin_edges, bin_weights):
    y = np.zeros_like(x)

    for w, (lo, hi) in zip(bin_weights, pairwise(bin_edges)):
        idx = (lo <= x) & (x < hi)
        y[idx] = w

    return y


def hist_err(x, weights, bin_edges, bin_weights):
    var = np.zeros_like(bin_weights)

    for i, (lo, hi) in enumerate(pairwise(bin_edges)):
        idx = (lo <= x) & (x < hi)
        var[i] = np.sum(weights[idx]**2)

    return np.sqrt(var)


def train_kde(x, w, bandwidth):
    # Fit a weighted KDE to the log-space M_c.
    kde = KDEUnivariate(x)
    kde.fit(weights=w, fft=False, bw=bandwidth)

    return kde


def chirp_mass_distribution(M_c, x_err, V, T, S, output_directory,
                            bandwidth="scott", n_smooth=1000):
    # Transform M_c into log-space.
    x = np.log10(M_c)
    # Compute the weights for each M_c.
    w = 1 / (V*T)

    # Generate `n_smooth` evenly-spaced values of log(M_c) for visualization
    # purposes.
    x_smooth = np.linspace(np.min(x), np.max(x), num=n_smooth)
    M_c_smooth = 10**x_smooth

    # Fit a histogram to the data.
    bin_weights, bin_edges = np.histogram(x, weights=w, bins=knuth_n_bins(x))
    bin_errors = hist_err(x, w, bin_edges, bin_weights)

    r_fn = lambda x: hist_pdf(x, bin_edges, bin_weights)
    r_err_fn = lambda x: hist_pdf(x, bin_edges, bin_errors)

    # Obtain discrete values of the histogram.
    r = r_fn(x)
    r_smooth = r_fn(x_smooth)
    # Obtain uncertainties
    r_err = r_err_fn(x)
    r_smooth_err = r_err_fn(x_smooth)
    # Compute the log of r
    log_r = np.log10(r)
    # Approximate uncertainty in log(r).
    log_r_err = 0.5 * (np.log10(r+r_err) - np.log10(r-r_err))

    # Determine which r's are zero, because they will be -inf in log-space
    idx = r != 0

    # Fit a power-law to the KDE, which is a linear fit in log-space.
    F = np.column_stack((np.ones_like(x[idx]), x[idx]))
    F_smooth = np.column_stack((np.ones_like(x_smooth), x_smooth))

    # Perform Bayesian linear regression.
    cov_r = np.diag(log_r_err[idx])
    icov_lambda = np.dot(F.T, np.linalg.solve(cov_r, F))
    mu_lambda = np.dot(np.linalg.solve(icov_lambda, F.T),
                       np.linalg.solve(cov_r, log_r[idx]))

    log_r_fit = np.dot(F_smooth, mu_lambda)
    cov_r_fit = np.dot(F_smooth, np.linalg.solve(icov_lambda, F_smooth.T))
    log_r_fit_err = np.sqrt(np.diag(cov_r_fit))

    r_fit = 10**log_r_fit
    r_fit_lower = 10**(log_r_fit - log_r_fit_err)
    r_fit_upper = 10**(log_r_fit + log_r_fit_err)


    ##############
    ## Plotting ##
    ##############

    # Create Figure.
    fig = plt.figure()
    # Set layout of Figure such that there are 3 vertically stacked subplots,
    # with the bottom one being 1/5 the size of the other two.
    gs = mpl.gridspec.GridSpec(2, 1, height_ratios=[5,1])
    # Create subplot axes, with shared x-axes.
    ax_pdf  = fig.add_subplot(gs[0])
    ax_data = fig.add_subplot(gs[1], sharex=ax_pdf)

    # Plot the histogram rate
    ax_pdf.plot(M_c_smooth, r_smooth, "b-")
    ax_pdf.fill_between(M_c_smooth, r_smooth-r_smooth_err, r_smooth+r_smooth_err,
                        color="b", alpha=0.1, edgecolor="b")
    #Plot the power-law fit to that KDE.
    ax_pdf.plot(M_c_smooth, r_fit, "r--")
    ax_pdf.fill_between(M_c_smooth, r_fit_lower, r_fit_upper,
                        color="r", alpha=0.1, edgecolor="r")
    # Plot the original data points, with random y-values for visualization
    # purposes.
    ax_data.scatter(M_c, np.random.uniform(size=np.shape(M_c)),
                    c="black", marker="+", alpha=0.2)

    # Make r(x) log-scale.
    ax_pdf.semilogx()
#    ax_pdf.loglog()
    ax_data.semilogx()

    # Set axis labels.
    ax_data.set_xlabel(r"$\mathcal{M}_c$")
    ax_pdf.set_ylabel(r"$r(\mathcal{M}_c)$")

    # Hide unwanted axis ticks and tick labels.
    plt.setp(ax_pdf.xaxis.get_ticklabels(), visible=False)
    ax_data.yaxis.set_ticks([])

    fig.savefig(path.join(output_directory, "chirp-mass-distribution.pdf"))


def dans_code(m_1, m_2, s, rho, q, q_err, eta, M_c, x_err, V,
              output_directory):
#    smoothing_poly_lnprior_example()

    chirp_mass_distribution(M_c, x_err, V, 1, 30, output_directory)
