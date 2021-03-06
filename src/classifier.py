"""

"""
from __future__ import division, print_function
import numpy as np

from matplotlib import pyplot as plt
from sklearn.svm import LinearSVC
from sklearn import grid_search
from astroML.utils import completeness_contamination

from os import path

import gw


def classifier(m_1, m_2, M_c, s,
               ax_pdf, ax_data, ax_log_pdf, ax_log_data,
               output_directory):
    M_c_front = M_c[:len(s)//2]
    M_c_end   = M_c[len(s)//2:]
    s_front   = s  [:len(s)//2]
    s_end     = s  [len(s)//2:]

    index_pos = (s==1)
    index_neg = (s==0)
    M_c_em = M_c[index_pos]
    M_c_not_em = M_c[index_neg]
    Mcem_max = max(M_c_em)
    Mcnotem_min = min(M_c_not_em)

    #Training a classifier with half the data set
    train = M_c[:len(M_c)//2]

    index_pos_half = index_pos[:len(s)//2]
    index_neg_half = index_neg[:len(s)//2]
    Mcem_half = train[index_pos_half]
    Mcnotem_half = train[index_neg_half]

    #Calculating the dividing line(using half the data)
    Mcem_half_max = max(Mcem_half)
    Mcem_half_min = min(Mcnotem_half)
    distance = abs(Mcem_half_max - Mcem_half_min)/2.
    line_half = Mcem_half_max + distance


    #Print the Max Mc of EM CP and Min of other along with the dividing line Mc
    print("It works" if Mcem_max < line_half < Mcnotem_min else "It doesn't work")
    print("The Minimum M_c for the Others is: ", Mcnotem_min)
    print("The Maximum M_c for the EM CP is: ", Mcem_max)
    print("The Dividing line trained by half the data is: ", line_half)

    for ax in [ax_pdf, ax_data, ax_log_pdf, ax_log_data]:
        ax.axvline(line_half, color="black", linestyle="--")

    fig_train, ax = plt.subplots()

    ax.scatter(M_c_front[s_front],
               np.random.uniform(0.0, 0.5, size=np.shape(M_c_front[s_front])),
               edgecolor="red", facecolor="none", marker="s")
    ax.scatter(M_c_end[s_end],
               np.random.uniform(0.5, 1.0, size=np.shape(M_c_end[s_end])),
               edgecolor="red", facecolor="red", marker="s")
    ax.scatter(M_c_front[~s_front],
               np.random.uniform(0.0, 0.5, size=np.shape(M_c_front[~s_front])),
               edgecolor="blue", facecolor="none", marker="o")
    ax.scatter(M_c_end[~s_end],
               np.random.uniform(0.5, 1.0, size=np.shape(M_c_end[~s_end])),
               edgecolor="blue", facecolor="blue", marker="o")

    ax.axvline(line_half, color="black", linestyle="--")

    ax.set_xlabel(r"$\mathcal{M}_c\ [M_\odot]$")

    ax.semilogx()
    ax.yaxis.set_ticklabels([])

    fig_train.savefig(path.join(output_directory, "classifier_comparison.pdf"))



    fig_2d, ax_2d = plt.subplots()

    m_1_smooth = np.logspace(0, 1.3, 1000)

    ax_2d.scatter(m_1[s], m_2[s],
                  color="red", marker="s")
    ax_2d.scatter(m_1[~s], m_2[~s],
                  color="blue", marker="o")

    ax_2d.plot(m_1_smooth, gw.m_2(m_1_smooth, line_half), "k--")

    ax_2d.set_xlabel(r"$m_1\ [M_\odot]$")
    ax_2d.set_ylabel(r"$m_2\ [M_\odot]$")

    ax_2d.loglog()

    fig_2d.savefig(path.join(output_directory, "mass-distribution.pdf"))
    

    m1_m2 = np.column_stack((m_1,m_2))
    train2 =  np.log10(m1_m2[:len(m1_m2)//2])
    clf = LinearSVC(C=100,class_weight='balanced').fit(train2, index_pos_half)
    index_pos_half_pred = clf.predict(train2)
    completeness2, contamination2 = completeness_contamination(index_pos_half_pred, index_pos_half)
    print("2D completeness: ", completeness2)
    print("2D contamination: ", contamination2)



    xx, yy = np.meshgrid(np.logspace(np.log10(m_1.min()), np.log10(m_1.max()), 500,endpoint=True),
                         np.logspace(np.log10(m_2.min()), np.log10(m_2.max()), 500,endpoint=True))

    Z = clf.predict(np.log10(np.c_[xx.ravel(), yy.ravel()]))
    
    Z = Z.reshape(xx.shape)
    print(np.unique(s))
    fig2d, ax2d = plt.subplots()
    ax2d.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8,antialiased=False,
                  extend='neither')
    ax2d.scatter(m_1, m_2, c=s, cmap=plt.cm.Paired)
    ax2d.set_xlabel('m$_1$')
    ax2d.set_ylabel('m$_2$')
    ax2d.loglog()
    ax2d.set_xlim(m_1.min(), m_1.max())
    ax2d.set_ylim(m_2.min(), m_2.max())
    
    fig2d.savefig(path.join(output_directory, "classifier-2D.pdf"))
