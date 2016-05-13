"""

"""
from __future__ import division, print_function

from astroML.density_estimation import knuth_bin_width
import numpy as np

from utils import pairwise



def knuth_n_bins(data):
    bandwidth = knuth_bin_width(data)
    return np.ceil((np.max(data) - np.min(data)) / bandwidth)



def hist_pdf(x, bin_edges, bin_weights):
    y = np.zeros_like(x)

    for w, (lo, hi) in zip(bin_weights, pairwise(bin_edges)):
        idx = (lo <= x) & (x < hi)
        y[idx] = w

    return y


def hist_err(x, weights, bin_edges, bin_weights):
    var = np.zeros_like(bin_weights)

    for i, (lo, hi) in enumerate(pairwise(bin_edges)):
        idx = (lo <= x) & (x < hi)
        var[i] = np.sum(weights[idx]**2)

    return np.sqrt(var)


def chirp_mass_distribution(M_c, M_c_smooth, x, x_smooth, w, s,
                            ax_pdf, ax_data):
    # Fit a histogram to the data.
    bin_weights, bin_edges = np.histogram(x, weights=w, bins=knuth_n_bins(x))
    bin_errors = hist_err(x, w, bin_edges, bin_weights)

    r_fn = lambda x: hist_pdf(x, bin_edges, bin_weights)
    r_err_fn = lambda x: hist_pdf(x, bin_edges, bin_errors)

    # Obtain discrete values of the histogram.
    r_smooth = r_fn(x_smooth)
    # Obtain uncertainties
    r_smooth_err = r_err_fn(x_smooth)


    # Plot the histogram rate
    ax_pdf.plot(M_c_smooth, r_smooth, "b-",
                label="Histogram")
    ax_pdf.fill_between(M_c_smooth, r_smooth-r_smooth_err, r_smooth+r_smooth_err,
                        color="b", alpha=0.1, edgecolor="b")
    # Plot the original data points, with random y-values for visualization
    # purposes.
    ax_data.scatter(M_c[s], np.random.uniform(size=np.shape(M_c[s])),
                    c="red", marker="+", alpha=1.0)
    ax_data.scatter(M_c[~s], np.random.uniform(size=np.shape(M_c[~s])),
                    c="blue", marker="x", alpha=0.1, edgecolor="none")

    return r_fn, r_err_fn
