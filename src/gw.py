from __future__ import division
import numpy as np


def mass_ratio(m_1, m_2):
    pass

def symmetric_mass_ratio(m_1, m_2):
    M = m_1 + m_2

    with np.errstate(invalid="ignore", divide="ignore"):
        return m_1 * m_2 * M**-2


def chirp_mass(m_1, m_2):
    M = m_1 + m_2
    eta = symmetric_mass_ratio(m_1, m_2)

    return eta**(3/5) * M
