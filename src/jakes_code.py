"""
Code written by Jacob Lange.
"""

from __future__ import division, print_function

import numpy as np
from matplotlib import pyplot as plt
from sklearn.svm import LinearSVC
from sklearn import grid_search
from astroML.utils import completeness_contamination

from os import path

import classifier

def jakes_code(m_1, m_2, s, rho, q, q_err, eta, M_c, M_c_err, V, output_directory):
    index_pos = (s==1)
    index_neg = (s==0)
    M_c_em = M_c[index_pos]
    M_c_not_em = M_c[index_neg]
    Mcem_max = max(M_c_em)
    Mcnotem_min = min(M_c_not_em)
#    print(Mcem_max)
#    print(Mcnotem_min)

    #Calculating where the dividing line should be
    dist = abs(Mcem_max - Mcnotem_min)/2.
    line = Mcem_max + dist
    print(line)
    #Figure of all the 5000 data points color coded with EM triggers and others
    fig, ax = plt.subplots(figsize=(10,10))

    ax.scatter(M_c_em, np.random.uniform(0, 1, len(M_c_em)),
               c='r', label='EM detection', edgecolors='none')
    ax.scatter(M_c_not_em, np.random.uniform(0, 1, len(M_c_not_em)),
               c='b', label='Others',edgecolors='none')

    ax.set_xlabel('$\mathcal{M}_{c}$')
    ax.set_xscale('log')
    ax.yaxis.set_ticklabels([])
    y = np.random.uniform( 0, 1, len(M_c_em))
#    print(y)
    line_array = np.empty(len(M_c_em))
    line_array.fill(line)
    fig.savefig(path.join(output_directory, "chirp-mass-classes.pdf"))

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
    
    #Showing half the data and the dividing line
    fig1, ax1 = plt.subplots(figsize=(10,10))

    ax1.scatter(Mcem_half, np.random.uniform(0, 1, len(Mcem_half)),
               c='r', label='EM detection',edgecolors='none')
    ax1.scatter(Mcnotem_half, np.random.uniform(0, 1, len(Mcnotem_half)),
               c='b', label='Others',edgecolors='none')

    ax1.set_xlabel('$\mathcal{M}_{c}$')
    ax1.set_xscale('log')

    ax1.yaxis.set_ticklabels([])
    
    y_half = np.random.uniform( 0, 1, len(Mcem_half))
    print(line_half)
    line_array_half = np.empty(len(Mcem_half))
    line_array_half.fill(line_half)
    ax1.axvline(line_half)
#    ax1.plot(line_array_half,y_half,'-')
    fig1.savefig(path.join(output_directory, "classifier_half.pdf"))

    #Showing the full data set with the dividing line trained by half the data
    fig2, ax2 = plt.subplots(figsize=(10,10))

    line_array_half_for_all = np.empty(len(M_c_em))
    line_array_half_for_all.fill(line_half)
    ax2.scatter(M_c_em, np.random.uniform(0, 1, len(M_c_em)),
                c='r', label='EM detection',edgecolors='none')
    ax2.scatter(M_c_not_em, np.random.uniform(0, 1,len(M_c_not_em)),
                c='b', label='Others',edgecolors='none')
    
    ax2.set_xlabel('$\mathcal{M}_{c}$')
    ax2.set_xscale('log')
    ax2.yaxis.set_ticklabels([])
#    ax2.plot(line_array_half_for_all,y,'-')
    ax2.axvline(line_half)
    fig2.savefig(path.join(output_directory, "classifier_all.pdf"))
    #Print the Max Mc of EM CP and Min of other along with the dividing line Mc
    print("It works" if Mcem_max < line_half < Mcnotem_min else "It doesn't work")
    print("The Minimum M_c for the Others is: ", Mcnotem_min)
    print("The Maximum M_c for the EM CP is: ", Mcem_max)
    print("The Dividing line trained by half the data is: ", line_half)

#    clf = LinearSVC(C=1000, class_weight='balanced')
#    clf.fit(train, index_pos_half)
#    s_pred_half = clf.predict(train)
    
#    M_c = M_c.reshape((-1,1))

#    print(*map(np.shape, [s_pred_half,index_pos_half]))

#    completeness, contamination = completeness_contamination(s_pred_half,
#                                                             index_pos_half)

#    print("completeness for half the data:", completeness)
#    print("contamination for half the data:" , contamination)

#    s_pred = clf.predict(M_c)

#    completeness, contamination = completeness_contamination(s_pred,
#                                                             index_pos)
#    print("completeness for the data:", completeness)
#    print("contamination for the data:" , contamination)
    
#    parameters = {'C':[1,1000]}
#    svr = LinearSVC(class_weight='balanced')
#    clf = grid_search.GridSearchCV(svr, parameters)
#    clf.fit(train, index_pos_half)
#    s_pred_half2 = clf.predict(train)
    
#    completeness, contamination = completeness_contamination(s_pred_half2,
#                                                            index_pos_half)

#    print("completeness for the data using CV:", completeness)
#    print("contamination for the data using CV:" , contamination)
