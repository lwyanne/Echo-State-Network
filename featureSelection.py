# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 13:40:01 2018
PCA
@author: acerr
"""

import numpy as np
from sklearn.decomposition import PCA
from function import *
path=os.path.abspath('.')
global path_data
path_data=os.path.join(path,'data')



#%%
samples=np.load(os.path.join(path_data,'40ms_samples_delta2.npy'))
teacher=np.load(os.path.join(path_data,'40ms_teacher_delta2.npy'))

samples[i,:]=normal(samples[i,:],mode=1)
pca=PCA()
pca.fit(samples)
print(pca.explained_variance_ratio_)
print(pca.singular_values_)  


#%%
samples=np.load(os.path.join(path_data,'25ms_samples_5_delta&delta2over8.npy'))
teacher=np.load(os.path.join(path_data,'25ms_teacher_5_delta&delta2over8.npy'))

for i in range(np.shape(samples)[1]):
    samples[:,i]=normal(samples[:,i],mode=1)
pca=PCA()
pca.fit(samples)
print(pca.explained_variance_ratio_)
print(pca.singular_values_)  