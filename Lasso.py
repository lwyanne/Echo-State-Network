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
samples=np.load(os.path.join(path_data,'samples.npy'))
teacher=np.load(os.path.join(path_data,'teacher.npy'))

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

i=3

teach1=dic_teach[i][0].reshape(-1,1)
teach2=dic_teach[i][1].reshape(-1,1)
model=LassoCV(max_iter=4000)
model.fit(dic_sam[i][0].T,teach1)

out=model.predict(dic_sam[i][1].T)
plt.plot(out,'.')
plt.plot(teach2)
esn=ESN(1,1)
err=esn.err(out,teach2,-1)        #0.32669
print(err)


