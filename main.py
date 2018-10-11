# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 22:20:33 2018
Compare to SVM
@author: acerr
"""

import random
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy
import scipy.linalg
from math import *
from sklearn import svm
import image
from mpl_toolkits.mplot3d import Axes3D
from function import *
from ESN import *
import os
import python_speech_features as sf
import librosa
import librosa.display
import matplotlib.gridspec as gridspec


i=3
n=900
c=0.3
sparsity=0.004
sparsity_in=0.004



# load the features
path=os.path.abspath('.')
global path_data
path_data=os.path.join(path,'data')
samples=np.load(os.path.join(path_data,'25ms_5samples_nodelta.npy'))
teacher=np.load(os.path.join(path_data,'25ms_5teacher_nodelta.npy'))
dim=np.shape(samples)[0]

#normalize
for j in range(dim):
    samples[j,:]=normal(samples[j,:],mode=1)
    
    
#divide samples into train-valid set and test set
tv_sam=samples[:,:-np.shape(samples)[-1]//11]
tv_teach=teacher[:,:-np.shape(samples)[-1]//11]
test_sam=samples[:,-np.shape(samples)[-1]//11:]
test_true=teacher[:,-np.shape(samples)[-1]//11:]

#divide train-valid set into train set and valid set seperately (subsequently use 10-fold cross-validation)
dic_sam=divide(tv_sam,10)
dic_teach=divide(tv_teach,10)


# choose some 9 fold to train, 1 fold to valid
train_sam=dic_sam[i][0]
train_tea=dic_teach[i][0]
valid_sam=dic_sam[i][1]
valid_tea=dic_teach[i][1]




# Initialize ESN
esn=ESN(n_inputs=dim,n_outputs=1,n_reservoir=n,spectral_radius=c,sparsity=sparsity,sparsity_in=sparsity_in)
esn.initweights()

esn.ridgeRegres(train_sam,train_tea,valid_sam,valid_tea)
np.save(os.path.join(path_data,'validESN_noDelta'),esn.outputs[0])


esn.update(test_sam[:,:100])
del esn.state
del esn.allstate
esn.update(test_sam[:,:],ifrestart=0)
esn.predict()
np.save(os.path.join(path_data,'testESN_noDelta'),esn.outputs[0])
err1=esn.err(esn.outputs,test_true,1)
err2=esn.err(esn.outputs,test_true,-1)
print('test NMSE==%f'%err1)
print('test error rate==%f'%err2)


plt.figure()
plt.plot(esn.outputs[0],'.',color='brown',label='output',markersize=1.2)
plt.plot(test_true[0],color='dimgray',label='true value',linewidth=0.8)