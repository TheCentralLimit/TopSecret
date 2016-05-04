"""
Code written by Jacob Lange.
"""

from __future__ import division, print_function

import numpy as np
from matplotlib import pyplot as plt
from sklearn.svm import LinearSVC
from astroML.utils import completeness_contamination

from os import path

import classifier

def jakes_code(m_1, m_2, s, rho, q, eta, M_c, V, output_directory):
    index_pos = (s==1)
    index_neg = (s==0)
    M_c_em = M_c[index_pos]
    M_c_not_em = M_c[index_neg]


    fig, ax = plt.subplots(figsize=(10,10))

    ax.scatter(M_c_em, np.random.uniform(0, 1, len(M_c_em)),
               c='r', label='EM detection')
    ax.scatter(M_c_not_em, np.random.uniform(0, 1, len(M_c_not_em)),
               c='b', label='Others')

    ax.set_xlabel('$\mathcal{M}_{c}$')

    ax.yaxis.set_ticklabels([])

    fig.savefig(path.join(output_directory, "chirp-mass-classes.pdf"))

    train = M_c[:len(M_c)//2]
    train = train.reshape((-1, 1))
    index_pos_half = index_pos[:len(s)//2]
    clf = LinearSVC(C=1000, class_weight='balanced')
    clf.fit(train, index_pos_half)
    s_pred_half = clf.predict(train)
    
    M_c = M_c.reshape((-1,1))

    print(*map(np.shape, [s_pred_half,index_pos_half]))

    completeness, contamination = completeness_contamination(s_pred_half,
                                                             index_pos_half)

    print("completeness for half the data:", completeness)
    print("contamination for half the data:" , contamination)

    s_pred = clf.predict(M_c)

    completeness, contamination = completeness_contamination(s_pred,
                                                             index_pos)
    print("completeness for the data:", completeness)
    print("contamination for the data:" , contamination)
