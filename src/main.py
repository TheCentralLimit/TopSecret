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


individual_code = {
    "chi"  : chis_code,
    "dan"  : dans_code,
    "jake" : jakes_code
}


def main(data_filename, person):
    m_1, m_2, s, rho = np.loadtxt(data_filename, unpack=True)
    q = gw.mass_ratio(m_1,m_2)
    eta = gw.symmetric_mass_ratio(m_1, m_2)
    M_c = gw.chirp_mass(m_1, m_2)
    D = gw.detectable_distance(M_c)
 
    individual_code[person.lower()](m_1, m_2, s, rho, q, eta, M_c)



if __name__ == "__main__":
    main(*argv[1:])
