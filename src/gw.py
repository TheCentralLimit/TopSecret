"""
Basic gravitational wave calculations.
"""
from __future__ import division, print_function
import numpy as np


def mass_ratio(m_1, m_2):
    """
    Returns the mass ratio q, defined such that the larger mass is in the
    numerator, and the smaller mass is in the denominator.
    """
    # Initialize array.
    q = np.empty_like(m_1)
    # Indices where m_1 or m_2 are bigger.
    m_1_bigger = m_1 > m_2
    m_2_bigger = ~m_1_bigger
    # Calculate the mass ratios for the two cases
    q[m_1_bigger] = m_1[m_1_bigger] / m_2[m_1_bigger]
    q[m_2_bigger] = m_2[m_2_bigger] / m_1[m_2_bigger]

    return q


def symmetric_mass_ratio(m_1, m_2):
    """
    Returns the symmetric mass ratio, eta.
    """
    M = m_1 + m_2

    with np.errstate(invalid="ignore", divide="ignore"):
        return m_1 * m_2 * M**-2


def chirp_mass(m_1, m_2):
    """
    Returns the chirp mass, M_c.
    """
    M = m_1 + m_2
    eta = symmetric_mass_ratio(m_1, m_2)

    return eta**(3/5) * M

def  detectable_distance(chirp_mass):
    """
    Returns the distance within which we can detect a system of a certain chirp
    mass unit [Mpc].
    """
    D = 200*(chirp_mass/1.2)**(5/6)

    return D
