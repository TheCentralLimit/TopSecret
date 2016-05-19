#!/usr/bin/env python2
from __future__ import division, print_function
from astroML.decorators import pickle_results
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy

from os import path
from sys import argv

from classifier import classifier
from fit import power_law
import gw
from density import chirp_mass_distribution
from utils import make_sure_path_exists

from chis_code import chis_code

def main(data_filename, output_directory, *features):
    # Set random seed.
    np.random.seed(1)
    # Create output directory if it does not exist.
    make_sure_path_exists(output_directory)
    # Load data from file.
    m_1, m_2, s, rho = np.loadtxt(data_filename, unpack=True)
    s = s.astype(bool)
    # Compute standard quantitites.
    eta = gw.symmetric_mass_ratio(m_1, m_2)
    M_c = gw.chirp_mass(m_1, m_2)
    x_err = gw.chirp_mass_log_error(M_c, rho)
    q = gw.mass_ratio(m_1, m_2)
    q_err = gw.mass_ratio_error(M_c, rho)
    D = gw.detectable_distance(M_c)
    V = (4/3) * np.pi * D**3
    T = 0.6
    # Transform M_c into log-space.
    x = np.log10(M_c)
    # Compute the weights for each M_c.
    w = 1 / (V*T)
    # Generate `n_smooth` evenly-spaced values of log(M_c) for visualization
    # purposes.
    x_smooth = np.linspace(np.min(x), np.max(x), num=1000)
    M_c_smooth = 10**x_smooth


    # Create Figure.
    fig_density = plt.figure()
    # Set layout of Figure such that there are 3 vertically stacked subplots,
    # with the bottom one being 1/5 the size of the other two.
    gs = mpl.gridspec.GridSpec(2, 1, height_ratios=[5,1])
    # Create subplot axes, with shared x-axes.
    ax_pdf  = fig_density.add_subplot(gs[0])
    ax_data = fig_density.add_subplot(gs[1], sharex=ax_pdf)

    # Set axis labels.
    ax_data.set_xlabel(r"$\mathcal{M}_c\ [M_\odot]$")
    ax_pdf.set_ylabel(r"$r(\mathcal{M}_c)$")

    # Hide unwanted axis ticks and tick labels.
    plt.setp(ax_pdf.xaxis.get_ticklabels(), visible=False)
    ax_data.yaxis.set_ticks([])

    ax_pdf.semilogx()
    ax_data.semilogx()


    # Create log-scale Figure.
    fig_log_density = plt.figure()
    # Set layout of Figure such that there are 3 vertically stacked subplots,
    # with the bottom one being 1/5 the size of the other two.
    gs = mpl.gridspec.GridSpec(2, 1, height_ratios=[5,1])
    # Create subplot axes, with shared x-axes.
    ax_log_pdf  = fig_log_density.add_subplot(gs[0])
    ax_log_data = fig_log_density.add_subplot(gs[1], sharex=ax_log_pdf)

    # Set axis labels.
    ax_log_data.set_xlabel(r"$\mathcal{M}_c\ [M_\odot]$")
    ax_log_pdf.set_ylabel(r"$r(\mathcal{M}_c)$")

    # Hide unwanted axis ticks and tick labels.
    plt.setp(ax_log_pdf.xaxis.get_ticklabels(), visible=False)
    ax_log_data.yaxis.set_ticks([])

    ax_log_pdf.loglog()
    ax_log_data.semilogx()


    r_fn, r_err_fn = chirp_mass_distribution(M_c, M_c_smooth, x, x_smooth, w, s,
                                             ax_pdf, ax_data,
                                             ax_log_pdf, ax_log_data)
    if ("power_law" in features) or ("all" in features):
        power_law(r_fn, r_err_fn, M_c, M_c_smooth, x, x_smooth,
                  ax_pdf, ax_data, ax_log_pdf, ax_log_data)
    if ("mcmc" in features) or ("all" in features):
        lam_mcmc = chis_code(np.log10(M_c),r_fn(np.log10(M_c)),r_err_fn(np.log10(M_c)),output_directory) # (x,y,yerr)
        

    if ("classifier" in features) or ("all" in features):
        classifier(m_1, m_2, M_c, s,
                   ax_pdf, ax_data, ax_log_pdf, ax_log_data,
                   output_directory)

    ax_pdf.legend()

    fig_density.savefig(path.join(output_directory,
                                  "chirp-mass-distribution.pdf"))
    fig_log_density.savefig(path.join(output_directory,
                                      "chirp-mass-log-distribution.pdf"))



if __name__ == "__main__":
    main(*argv[1:])
