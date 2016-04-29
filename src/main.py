#!/usr/bin/env python2
from __future__ import division, print_function
from astroML.decorators import pickle_results
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy

from sys import argv

import gw

from chis_code import chis_code
from dans_code import dans_code
from jakes_code import jakes_code

def main(data_filename):
    m_1, m_2, s, rho = np.loadtxt(data_filename, unpack=True)
    eta = gw.symmetric_mass_ratio(m_1, m_2)
    M_c = gw.chirp_mass(m_1, m_2)

    print(np.min(eta), np.max(eta), np.mean(eta))
    print(np.min(M_c), np.max(M_c), np.mean(M_c))

    q = gw.mass_ratio(m_1,m_2)
    chis_code(m_1,m_2,s,rho,q,eta,M_c)
    dans_code(m_1,m_2,s,rho,q,eta,M_c)
    jakes_code(m_1,m_2,s,rho,q,eta,M_c)
    

if __name__ == "__main__":
    main(*argv[1:])
