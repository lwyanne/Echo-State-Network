# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 23:01:01 2018

@author: acerr
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 16:56:18 2018

@author: acerr
"""


import random
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy
import scipy.linalg
from math import *
import image
from mpl_toolkits.mplot3d import Axes3D
from function import *
from ESN import *
import os
import python_speech_features as sf
import librosa
import librosa.display
import matplotlib.gridspec as gridspec


def esn(trsig,tarsig,testsig,real,n_inputs=26,n_reservoir=200,n_outputs=1,a=1,b=1,c=1,sparsity=0.005,sparsity_in=0.006,ifload=0):
    if not ifload:
        network=ESN(n_inputs,n_outputs,n_reservoir,c,sparsity,sparsity_in,b=b,a=a,noise=0,seednum=2)
        network.initweights()
        print(network.a)
        np.save(os.path.join(path_data,'V'),network.V)
        np.save(os.path.join(path_data,'W'),network.W)
        np.save(os.path.join(path_data,'Wb'),network.Wb)
        np.save(os.path.join(path_data,'noiseVec'),network.noiseVec)
        
    else:
        network=ESN(n_inputs,n_outputs,n_reservoir,c,sparsity,sparsity_in,b=b,a=a)
        network.V=np.load(os.path.join(path_data,'V.npy'))
        network.Wb=np.load(os.path.join(path_data,'Wb.npy'))
        network.W=np.load(os.path.join(path_data,'W.npy'))
        network.noiseVec=np.load(os.path.join(path_data,'noiseVec.npy'))
    print('successfully initialized weights matrix\n\n')
    network.update(trsig[:,:64],1)
    del network.allstate
    network.update(trsig,0)
    del trsig
    network.fit(network.allstate,tarsig,0)
    network.predict()
    print('training error==',network.err(network.outputs,tarsig,-1))
    print('training error==',network.err(network.outputs,tarsig,1))
    del network.allstate
    network.update(testsig[:,:64],1)
    del network.allstate
    network.update(testsig,0)
    network.predict()
    del testsig
    error=network.err(network.outputs,real,-1)
    print('successfully train the readout matrix\n\n ',np.shape(real))
    temp=network.outputs
    del network
    return error,temp

# load the features

path=os.path.abspath('.')
global path_data
path_data=os.path.join(path,'data')
samples=np.load(os.path.join(path_data,'40ms_samples_51_delta.npy'))
teacher=np.load(os.path.join(path_data,'40ms_teacher_51_delta.npy'))



#normalize
for i in range(26):
    samples[i,:]=normal(samples[i,:],mode=1)



divideNum=60
#divide into train-valid set and test set
tv_sam=samples[:,:-np.shape(samples)[-1]//2]
tv_teach=teacher[:,:-np.shape(samples)[-1]//2]
test_sam=samples[:,-np.shape(samples)[-1]//2:]
test_true=teacher[:,-np.shape(samples)[-1]//2:]










matrix=[]

dic_sam=divide(tv_sam,divideNum)
dic_teach=divide(tv_teach,divideNum)
error=[]
i=4# i=0,1,2,....,9
ifload=0

c=1
n=100
a,b=esn(dic_sam[i][0],dic_teach[i][0],dic_sam[i][1],dic_teach[i][1],n_inputs=26,n_reservoir=n,sparsity=1,sparsity_in=1,a=1,c=c,ifload=ifload)
error.append(a)
print('c==%f, spasity==%f,error==%f'%(c,n,a))






