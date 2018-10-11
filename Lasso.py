# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 15:39:36 2018

@author: acerr
"""

# -*- coding: utf-8 -*-
"""
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
from sklearn.linear_model import LassoCV


path=os.path.abspath('.')
global path_data
path_data=os.path.join(path,'data')
samples=np.load(os.path.join(path_data,'25ms_5samples.npy'))
teacher=np.load(os.path.join(path_data,'25ms_5teacher.npy'))

for i in range(26):
    samples[i,:]=normal(samples[i,:],mode=1)


#divide into train-valid set and test set
tv_sam=samples[:,:-np.shape(samples)[-1]//11]
tv_teach=teacher[:,:-np.shape(samples)[-1]//11]
test_sam=samples[:,-np.shape(samples)[-1]//11:]
test_true=teacher[:,-np.shape(samples)[-1]//11:]

matrix=[]

dic_sam=divide(tv_sam,10)
dic_teach=divide(tv_teach,10)

i=3

train_tea=dic_teach[i][0].reshape(-1,1)
valid_tea=dic_teach[i][1].reshape(-1,1)
model=LassoCV(max_iter=4000)
model.fit(dic_sam[i][0].T,train_tea)
out_train=model.predict(dic_sam[i][0].T)
out_valid=model.predict(dic_sam[i][1].T)
np.save(os.path.join(path_data,'validLasso'),out_valid)


out_test=model.predict(test_sam.T)
np.save(os.path.join(path_data,'testLasso'),out_test)

plt.figure()
plt.plot(out_valid,'.')
plt.plot(valid_tea)
esn=ESN(1,1)
err=esn.err(out_valid,valid_tea,-1)        #0.32669
err1=esn.err(out_train,train_tea,-1)
err2=esn.err(out_test,test_true,-1)
print(err)


