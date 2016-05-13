"""

"""
from __future__ import division, print_function
import numpy as np

from matplotlib import pyplot as plt
from sklearn.svm import LinearSVC
from sklearn import grid_search
from astroML.utils import completeness_contamination

from os import path




def classifier(M_c, s, ax_pdf, ax_data, output_directory):
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

    ax_pdf.axvline(line_half, color="black", linestyle="--")
    ax_data.axvline(line_half, color="black", linestyle="--")

    fig, ax = plt.subplots()

    ax.scatter(M_c_em,
               np.random.uniform(0.0, 0.5, size=np.shape(M_c_em)),
               edgecolor="red", facecolor="none", marker="s")
    ax.scatter(Mcem_half,
               np.random.uniform(0.5, 1.0, size=np.shape(Mcem_half)),
               edgecolor="red", facecolor="red", marker="s")
    ax.scatter(M_c_not_em,
               np.random.uniform(0.0, 0.5, size=np.shape(M_c_not_em)),
               edgecolor="blue", facecolor="none", marker="o")
    ax.scatter(Mcnotem_half,
               np.random.uniform(0.5, 1.0, size=np.shape(Mcnotem_half)),
               edgecolor="blue", facecolor="blue", marker="o")

    ax.axvline(line_half, color="black", linestyle="--")

    ax.set_xlabel(r"$\mathcal{M}_c\ [M_\odot]$")

    ax.semilogx()
    ax.yaxis.set_ticklabels([])

    fig.savefig(path.join(output_directory, "classifier_comparison.pdf"))
