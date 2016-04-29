from __future__ import division, print_function
from density import density_estimator



def dans_code(m_1 , m_2, s, rho, q, eta, M_c, V):
    kde = density_estimator(M_c, V, 1)

    print(kde.cdf)
