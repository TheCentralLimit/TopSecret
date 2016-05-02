"""
Code written by Daniel Wysocki.
"""

from __future__ import division, print_function

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from statsmodels.nonparametric.kde import KDEUnivariate

from os import path


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


def dans_code(m_1, m_2, s, rho, q, eta, M_c, V, output_directory):
    chirp_mass_distribution(M_c, V, 1, output_directory)
