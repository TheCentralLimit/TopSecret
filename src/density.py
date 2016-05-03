from __future__ import division, print_function
import numpy as np
from statsmodels.nonparametric.kde import KDEUnivariate


def density_estimator(M_c, V, T, **kwargs):
    x = np.log10(M_c)
    w = 1 / (V*T)

    kde = KDEUnivariate(x)
    kde.fit(weights=w, fft=False, **kwargs)

    return kde # placeholder, return something different in the future

# Chi: calculate r from M_c. Can merge with kde later.
def formation_rate_estimator(M_c,V,T,bandwidth="scott"):
    x = np.log10(M_c)
    # Compute the weights for each M_c.
    w = 1 / (V*T)

    # Fit a weighted KDE to the log-space M_c.
    kde = KDEUnivariate(x)
    kde.fit(weights=w, fft=False, bw=bandwidth)

    # Evaluate the KDE at the observed log(M_c) values.
    r = kde.evaluate(x)
    
    return r