# -*- coding: utf-8 -*-
"""
SVM


Created on Sun Oct  7 15:18:26 2018

@author: acerr
"""


from sklearn import svm
from function import *
from ESN import *
import librosa.display
#load the datas
i=3
path=os.path.abspath('.')
global path_data
path_data=os.path.join(path,'data')
samples=np.load(os.path.join(path_data,'25ms_5samples.npy'))
teacher=np.load(os.path.join(path_data,'25ms_5teacher.npy'))
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



X=train_sam.T
y=train_tea[0]
clf=svm.SVC(gamma='auto')
clf.fit(X,y)
z=clf.predict(test_sam.T)
z0=clf.predict(train_sam.T)

fig1 = plt.figure()
gridspec.GridSpec(8,8)
plt.subplot2grid((8,8),(0,0),7,8)
z=np.reshape(z,(1,np.size(z)))
librosa.display.specshow(z)
plt.colorbar()
plt.subplot2grid((8,8),(7,0),1,8)
librosa.display.specshow(test_true)
plt.xlabel('time')
plt.colorbar()

np.save(os.path.join(path_data,'testSVM'),z)

z1=clf.predict(valid_sam.T)
fig1 = plt.figure()
gridspec.GridSpec(8,8)
plt.subplot2grid((8,8),(0,0),7,8)
z1=np.reshape(z1,(1,np.size(z1)))
librosa.display.specshow(z1)
plt.colorbar()
plt.subplot2grid((8,8),(7,0),1,8)
librosa.display.specshow(valid_tea)
plt.xlabel('time')
plt.colorbar()
np.save(os.path.join(path_data,'validSVM'),z1)
