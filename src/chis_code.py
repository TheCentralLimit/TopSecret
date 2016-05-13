# -*- coding: utf-8 -*-
"""
Code written by Chi Nguyen.
"""

from __future__ import division, print_function
from numpy.polynomial import polynomial as P
from os import path
#import matplotlib

import emcee
import corner
import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as pl
from matplotlib.ticker import MaxNLocator
#matplotlib.use('Agg')

# group modulus
import density as ds
import mcmc

# Reproducible results!
np.random.seed(123)

def chis_code(x,y,yerr,output_directory):
    # Transform M_c into log-space.
    #index = check_nonzero(y_in)
    #x = x_in[index>0]
    #y_not_log = y_in[index>0]
    #y = np.log10(y_not_log)
    #yerr_not_log = yerr_in[index>0]
    #yerr = error(y_not_log,yerr_not_log)
    #yerr = np.log10(yerr_not_log)
    
    print(len(yerr))
    
    degree = 5 # degree of polynomial
    # least square fitting
    lam_ls = mcmc.least_square(x,y,yerr,degree,output_directory)
    lam_ml = mcmc.maximum_likelihood(x,y,yerr,degree,lam_ls,output_directory)
    lam_mcmc = mcmc.MCMC(x,y,yerr,degree,lam_ml,output_directory)
    
    return lam_mcmc

def check_nonzero(y_in):
    index = (y_in[:]>0).astype(int)

    return index


def error(y,yerr):
    yerr_return = (np.log10(y+yerr) - np.log10(y-yerr))/2.0
    index = (yerr_return == 0).astype(int)
    print(np.sum(index))
    yerr_return[index>0] = 10**(-8.0)
    return yerr_return
    
