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
from utils import make_sure_path_exists

from chis_code import chis_code
from dans_code import dans_code
from jakes_code import jakes_code


individual_code = {
    "chi"  : chis_code,
    "dan"  : dans_code,
    "jake" : jakes_code
}


def individual_fn(name, output_directory):
    # Ignore case in name.
    name = name.lower()
    # Make sure name is valid.
    assert individual_code.get(name) is not None, "Invalid name: " + name
    # Output to sub-directory [output_directory]/[name]
    output_directory = path.join(output_directory, name)
    # Create output directory if it does not exist.
    make_sure_path_exists(output_directory)

    return lambda *args: individual_code[name](*(args + (output_directory,)))


def main(data_filename, output_directory, name):
    # Set random seed.
    np.random.seed(1)
    # Create output directory if it does not exist.
    make_sure_path_exists(output_directory)
    # Load data from file.
    m_1, m_2, s, rho = np.loadtxt(data_filename, unpack=True)
    # Compute standard quantitites.
    eta = gw.symmetric_mass_ratio(m_1, m_2)
    M_c = gw.chirp_mass(m_1, m_2)
    M_c_err = gw.chirp_mass_error(M_c, rho)
    q = gw.mass_ratio(m_1, m_2)
    q_err = gw.mass_ratio_error(M_c, rho)
    D = gw.detectable_distance(M_c)
    V = (4/3) * np.pi * D**3
    # Run the code written by an individual person.
    fn = individual_fn(name, output_directory)
    fn(m_1, m_2, s, rho, q, q_err, eta, M_c, M_c_err, V)



if __name__ == "__main__":
    main(*argv[1:])
