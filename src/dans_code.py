from __future__ import division, print_function
from density import density_estimator

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt



def chirp_mass_distribution(M_c, V, T):
    kde = density_estimator(M_c, V, 1, kernel="gau", bw="scott")

    x_smooth = np.linspace(0, np.log10(np.max(M_c)),
                           num=1000)

    r = kde.evaluate(x_smooth)

    fig, (ax_pdf, ax_cdf) = plt.subplots(2)

    ax_pdf.plot(x_smooth, r)

    ax_pdf.semilogy()


    ax_pdf.set_xlabel(r"$x$")
    ax_pdf.set_ylabel(r"$\hat{r}$")

    ax_cdf.plot(kde.cdf)

    plt.show()


def dans_code(m_1 , m_2, s, rho, q, eta, M_c, V):
    chirp_mass_distribution(M_c, V, 1)
