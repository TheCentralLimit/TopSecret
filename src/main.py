#!/usr/bin/env python2
from __future__ import division, print_function
from astroML.decorators import pickle_results
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy

from os import path
from sys import argv

import gw
from density import chirp_mass_distribution
from utils import make_sure_path_exists



def main(data_filename, output_directory):
    # Set random seed.
    np.random.seed(1)
    # Create output directory if it does not exist.
    make_sure_path_exists(output_directory)
    # Load data from file.
    m_1, m_2, s, rho = np.loadtxt(data_filename, unpack=True)
    # Compute standard quantitites.
    eta = gw.symmetric_mass_ratio(m_1, m_2)
    M_c = gw.chirp_mass(m_1, m_2)
    x_err = gw.chirp_mass_log_error(M_c, rho)
    q = gw.mass_ratio(m_1, m_2)
    q_err = gw.mass_ratio_error(M_c, rho)
    D = gw.detectable_distance(M_c)
    V = (4/3) * np.pi * D**3
    T = 0.6

    # Create Figure.
    fig_density = plt.figure()
    # Set layout of Figure such that there are 3 vertically stacked subplots,
    # with the bottom one being 1/5 the size of the other two.
    gs = mpl.gridspec.GridSpec(2, 1, height_ratios=[5,1])
    # Create subplot axes, with shared x-axes.
    ax_pdf  = fig_density.add_subplot(gs[0])
    ax_data = fig_density.add_subplot(gs[1], sharex=ax_pdf)

    # Set axis labels.
    ax_data.set_xlabel(r"$\mathcal{M}_c$")
    ax_pdf.set_ylabel(r"$r(\mathcal{M}_c)$")

    # Hide unwanted axis ticks and tick labels.
    plt.setp(ax_pdf.xaxis.get_ticklabels(), visible=False)
    ax_data.yaxis.set_ticks([])

    ax_pdf.semilogx()
    ax_data.semilogx()


    r_fn, r_err_fn = chirp_mass_distribution(M_c, V, T,
                                             ax_pdf, ax_data,
                                             n_smooth=1000)

    fig_density.savefig(path.join(output_directory,
                                  "chirp-mass-distribution.pdf"))



if __name__ == "__main__":
    main(*argv[1:])
