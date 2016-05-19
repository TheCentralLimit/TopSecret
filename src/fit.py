"""

"""
from __future__ import division, print_function

import numpy as np


def power_law(r_fn, r_err_fn, M_c, M_c_smooth, x, x_smooth,
              ax_pdf, ax_data, ax_log_pdf, ax_log_data):
    # Obtain discrete values of the histogram.
    r = r_fn(x)
    # Obtain uncertainties
    r_err = r_err_fn(x)
    # Compute the log of r
    log_r = np.log10(r)

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

    # Plot the power-law fit to that KDE.
    for ax in [ax_pdf, ax_log_pdf]:
        ax.plot(M_c_smooth, r_fit, "r--",
                label="Power law")
        ax.fill_between(M_c_smooth, r_fit_lower, r_fit_upper,
                        color="r", alpha=0.1, edgecolor="r")
