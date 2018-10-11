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
import math
u1=Lorenz(len=100000)
u1.downsample(10)
u1.normalize()
u1=u1.get()

u2=Lorenz(init_point=(1,1,3),len=100000)
u2.downsample(10)
u2.normalize()
u2=u2.get()


def timeshift(dt,i):
    """
    i is used to subplot
    if i==-1, then no figure would be plotted
    """

    u0=u1[0][22:-22-dt]
    utarget=u1[0][22+dt:-22]
     # print('utarget===',utarget)
     # plt.plot(u0)
     # plt.plot(utarget)
    

    utest=u2[0][22:-22-dt]
    utest_target=u2[0][22+dt:-22]
    
    
    print('dt=======',dt)
    print('lenth=============',len(u0))
    esn.update(u0[:100],1)
    del esn.allstate
    esn.update(u0,0)
    esn.fit(u0,utarget,0)
    esn.predict()
    print('training error==',esn.err(esn.outputs[0],utarget,1))
    del esn.allstate
    esn.update(utest[:100],1)
    del esn.allstate
    esn.update(utest,0)
    esn.predict()
    er=esn.err(esn.outputs[0],utest_target,1)
    
    if i<0: return er
    if i%4==0:plt.figure()
    t=np.arange(0,esn.siglenth)
    plt.subplot(2,2,i%4+1)
    plt.title('timeshift=%d, log10(error)=%.3f'%(dt,math.log10(er)))
    plt.plot(esn.outputs[0],'.',color='brown',label='output',markersize=1.2)
    plt.plot(utest_target,color='dimgray',label='true value',linewidth=0.8)
    plt.legend()
    return er

#%%err=[]
esn=ESN(n_inputs=1,n_outputs=1,sparsity=0.05,ifplot=1)
esn.initweights()
i=0
err=[]
for dt in np.arange(-10,11):

     err.append(timeshift(dt,i))
     i+=1
     print(err)
#err=[295.60912302597865,245.18144723228568, 10.888772547097474, 15.412096058138188, 8.1037759534482987, 3.8312916015223508, 0.50188617288938764, 1.2340219229662961, 0.18434224085513634, 0.33480990479688671, 0.21793509382861573, 1.2868466072348357, 4.3514284124729361, 21.457340263785259, 422.39221136781953, 805.34487350417078, 194.21032430960574, 512.90323182697409, 2925.915592466285, 2484.8587280908305, 4251.6436440069592]
err=list(map(math.log,err))
plt.figure()
plt.plot(err)


# print(timeshift(0))
plt.show()
#%% timeshift error figure with different randomseeds.
err=np.zeros((10,41))
for seed in range(20,30):
    esn=ESN(n_inputs=1,n_outputs=1,n_reservoir=20,sparsity=0.5,a=0,sparsity_in=1,ifplot=1,seednum=seed)
    esn.initweights()
    for dt in np.arange(-20,21):
        temp=(timeshift(dt,-1))
        err[seed-20][dt+20]=temp
        print('seed=%d, dt=%d, error=%f'%(seed,dt,temp))

error=np.mean(err,axis=0)
plt.figure()
plt.plot(np.arange(-20,21),np.log10(error))
plt.ylabel('log10(error)')
plt.xlabel('timeshift')
#TODO: 用此刻输入和非此刻输入的问题！！




#%% Linear Reservoir
err=np.zeros((10,41))
for seed in range(20,30):
    esn=LESN(n_inputs=1,n_outputs=1,n_reservoir=20,sparsity=0.5,sparsity_in=1,ifplot=0)
    esn.initweights()
    for dt in np.arange(-20,21):
        temp=(timeshift(dt,1))
        err[seed-20][dt+20]=temp
        print('seed=%d, dt=%d, error=%f'%(seed,dt,temp))

error=np.mean(err,axis=0)
plt.figure()
plt.plot(np.arange(-20,21),np.log10(error))
plt.ylabel('log10(error)')
plt.xlabel('timeshift')


"""
Linear1.npy: n_reservoir=20, sparsity=0.5, sparsity_in=1,c=0.95

"""




#%% Timeshift=-5  Nonlinear Regarding a

err=np.zeros((10,10))
i=0
for seed in range(20,30):
    for a in np.arange(0,0.2,0.02):
        esn=ESN(n_inputs=1,n_outputs=1,n_reservoir=20,sparsity=0.5,sparsity_in=1,ifplot=1,a=a,seednum=seed)
        esn.initweights()

        temp=(timeshift(-5,-1))
        err[seed-20][i%10]=temp
        i+=1
        print('seed=%d, a=%f, error=%f'%(seed,a,temp))

error_a=np.mean(err,axis=0)
plt.figure()
plt.plot(np.arange(0,0.2,0.02),np.log10(error_a))
plt.ylabel('log10(error)')
plt.xlabel('timeshift')


#%% Timeshift=-5  Nonlinear Regarding a


i=0
l=[5,10,15,20,25,30,35,40,45,50,55,60]
err=np.zeros((10,len(l)))
for seed in range(20,30):
    for n in l:
        esn=ESN(n_inputs=1,n_outputs=1,n_reservoir=n,sparsity=5/n,sparsity_in=1,ifplot=1,a=0.02,seednum=seed)
        esn.initweights()

        temp=(timeshift(-5,-1))
        err[seed-20][i%len(l)]=temp
        i+=1
        print('seed=%d, a=%f, error=%f'%(seed,a,temp))

error_a=np.mean(err,axis=0)
plt.figure()
plt.plot(l,np.log10(error_a))
plt.ylabel('log10(error)')
plt.xlabel('timeshift')