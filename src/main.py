#!/usr/bin/env python2
from __future__ import division, print_function
from astroML.decorators import pickle_results
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy

from sys import argv

import gw


def main(data_filename):
    m_1, m_2, s, rho = np.loadtxt(data_filename, unpack=True)
    eta = gw.symmetric_mass_ratio(m_1, m_2)
    M_c = gw.chirp_mass(m_1, m_2)

    print(np.min(eta), np.max(eta), np.mean(eta))
    print(np.min(M_c), np.max(M_c), np.mean(M_c))


if __name__ == "__main__":
    main(*argv[1:])
