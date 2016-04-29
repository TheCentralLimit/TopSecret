from __future__ import division, print_function
import numpy as np
from statsmodels.nonparametric.kde import KDEUnivariate


def density_estimator(M_c, V, T, **kwargs):
    x = np.log10(M_c)
    w = 1 / (V*T)

    kde = KDEUnivariate(x)
    kde.fit(weights=w, fft=False, **kwargs)

    return kde # placeholder, return something different in the future
