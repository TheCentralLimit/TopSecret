"""
Code written by Daniel Wysocki.
"""

from __future__ import division, print_function

import numpy as np
from numpy.polynomial import polynomial as P
import matplotlib as mpl
import matplotlib.pyplot as plt
from statsmodels.nonparametric.kde import KDEUnivariate

from os import path


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



def chirp_mass_distribution(M_c, V, T, output_directory,
                            bandwidth="scott", n_smooth=1000):
    # Transform M_c into log-space.
    x = np.log10(M_c)
    # Compute the weights for each M_c.
    w = 1 / (V*T)

    # Fit a weighted KDE to the log-space M_c.
    kde = KDEUnivariate(x)
    kde.fit(weights=w, fft=False, bw=bandwidth)

    # Evaluate the KDE at the observed log(M_c) values.
    r = kde.evaluate(x)

    # Generate `n_smooth` evenly-spaced values of log(M_c) for visualization
    # purposes.
    x_smooth = np.linspace(np.min(x), np.max(x), num=n_smooth)
    # Evaluate the KDE at the evenly-spaced values of log(M_c).
    r_smooth = kde.evaluate(x_smooth)

    # Fit a power-law to the KDE, which is a linear fit in log-space.
    design_matrix = np.column_stack((np.ones_like(x), x))
    intercept, slope = np.linalg.lstsq(design_matrix, np.log10(r))[0]
    r_fit = 10**(slope*x_smooth + intercept)

    ##############
    ## Plotting ##
    ##############

    # Create Figure.
    fig = plt.figure()
    # Set layout of Figure such that there are 3 vertically stacked subplots,
    # with the bottom one being 1/5 the size of the other two.
    gs = mpl.gridspec.GridSpec(3, 1, height_ratios=[5,5,1])
    # Create subplot axes, with shared x-axes.
    ax_pdf  = fig.add_subplot(gs[0])
    ax_cdf  = fig.add_subplot(gs[1], sharex=ax_pdf)
    ax_data = fig.add_subplot(gs[2], sharex=ax_pdf)

    # Plot the KDE rate, as well as the power-law fit to that KDE.
    ax_pdf.plot(x_smooth, r_smooth)
    ax_pdf.plot(x_smooth, r_fit, "r--")
    # Plot the CDF of the KDE.
    ax_cdf.plot(np.sort(x), kde.cdf)
    # Plot the original data points, with random y-values for visualization
    # purposes.
    ax_data.scatter(x, np.random.uniform(size=np.shape(x)),
                    c="black", marker="+", alpha=0.2)

    # Make r(x) log-scale.
    ax_pdf.semilogy()

    # Set axis labels.
    ax_data.set_xlabel(r"$x$")
    ax_pdf.set_ylabel(r"$r(x)$")
    ax_cdf.set_ylabel(r"CDF")

    # Hide unwanted axis ticks and tick labels.
    plt.setp(ax_pdf.xaxis.get_ticklabels(), visible=False)
    plt.setp(ax_cdf.xaxis.get_ticklabels(), visible=False)
    ax_data.yaxis.set_ticks([])

    fig.savefig(path.join(output_directory, "chirp-mass-distribution.pdf"))


def dans_code(m_1, m_2, s, rho, q, q_err, eta, M_c, M_c_err, V,
              output_directory):
    print(np.column_stack((M_c, M_c_err, q, q_err)))

    smoothing_poly_lnprior_example()

    chirp_mass_distribution(M_c, V, 1, output_directory)
