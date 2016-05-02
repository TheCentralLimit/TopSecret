from __future__ import division, print_function
import numpy as np
from matplotlib import pyplot as plt
import classifier
from sklearn.svm import LinearSVC
from astroML.utils import completeness_contamination

def jakes_code(m_1,m_2,s,rho,q,eta,M_c,V):
    fig, ax = plt.subplots(figsize=(10,10))
    index_pos=(s==1)
    index_neg=(s==0)
    M_c_em=M_c[index_pos]
    M_c_not_em=M_c[index_neg]
    ax.scatter(M_c_em,np.random.uniform(0,1,len(M_c_em)),c='g',label='EM detection')
    ax.scatter(M_c_not_em,np.random.uniform(0,1,len(M_c_not_em)),c='b',label='Others')
    ax.set_ylabel('random.uniform')
    ax.set_xlabel('$M_{c}$')
#    plt.show()
    train=M_c[:len(M_c)/2]
    print(train)
    print(M_c[2499])
    train=train.reshape(-1,1)
    index_pos_half=index_pos[:len(s)/2]
    clf=LinearSVC(C=10,class_weight='balanced')
    clf.fit(train,index_pos_half)
    s_pred=clf.predict(train)
    completeness, contamination=completeness_contamination(s_pred,index_pos_half)
    print("completeness:", completeness)
    print("contamination:" , contamination)
