# -*- coding: utf-8 -*-
"""
in



Created on Mon Aug 13 17:38:08 2018

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
    error,numda=network.ridgeRegres(trsig,tarsig,testsig,real,path_data,ifload=ifload)
    print('successfully train the readout matrix\n\n ',np.shape(real))
    temp=network.outputs
    del network
    return error,temp

# load the features

path=os.path.abspath('.')
global path_data
path_data=os.path.join(path,'data')
samples=np.load(os.path.join(path_data,'40ms_samples_delta2.npy'))
teacher=np.load(os.path.join(path_data,'40ms_teacher_delta2.npy'))



#normalize
for i in range(26):
    samples[i,:]=normal(samples[i,:],mode=1)


#divide into train-valid set and test set
tv_sam=samples[:,:-np.shape(samples)[-1]//11]
tv_teach=teacher[:,:-np.shape(samples)[-1]//11]
test_sam=samples[:,-np.shape(samples)[-1]//11:]
test_true=teacher[:,-np.shape(samples)[-1]//11:]










matrix=[]

dic_sam=divide(tv_sam)
dic_teach=divide(tv_teach)
error=[]
i=3# i=0,1,2,....,9
ifload=0
#for t in np.linspace(-0.1,0.2,3):
#for c in np.linspace(0.1,1.1,9):
for c in [0.1]:
    #for spas in [0.1,0.2,0.3,0.4,0.5]:
    for spas in [0.3]:
        a,b=esn(dic_sam[i][0],dic_teach[i][0],dic_sam[i][1],dic_teach[i][1],n_inputs=39,n_reservoir=20,sparsity=spas,sparsity_in=spas,a=1,c=c,ifload=ifload)
        del b
        error.append(a)
        print('c==%f, spasity==%f,error==%f'%(c,spas,a))
    matrix.append(error)
    del error
    error=[]

#np.save(os.path.join(path_data,'candsparsity——n20'),matrix) 
matrix=np.array(matrix)
fig=plt.figure()
x=np.linspace(0,1,15)
y=[0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01,0.012,0.015,0.1]
x,y=np.meshgrid(x,y)
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(x,y,matrix.T,rstride=1,cstride=1)
plt.show()
#a,b=esn(dic_sam[i][0],dic_teach[i][0],dic_sam[i][1],dic_teach[i][1],n_inputs=26,n_reservoir=1000,sparsity=0.01,sparsity_in=0.01,a=1,c=0.3,ifload=ifload)
#error.append(a)


ifload+=1
