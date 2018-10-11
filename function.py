"""
all the functions used in this assignment
"""
import os
import re
import random
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import librosa
import python_speech_features as sf
import scipy
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
from math import *
import image
from mpl_toolkits.mplot3d import Axes3D



path=os.path.abspath('.')
global path_data
path_data=os.path.join(path,'data')


def discard(sig):
    sig=np.array(sig)
    l_transient=min(int(np.shape(sig)[-1] / 10), 100)
    if sig.ndim==2:
        temp=sig[:,l_transient:]
    else:
        temp=sig[l_transient:]
    del sig
    return temp

def normal(u,mode):
    """
    This is used to normalize u
    """
    if mode==1:
        l=len(u)
        temp=[]
        mean=sum(u)/l
        var=sqrt(sum([(x-mean)**2 for x in u])/l)
        for i in u:
            i=(i-mean)/var
            temp.append(i)
    else:
        l=len(u)
        temp=[]
        d=max(u)-min(u)
        m=min(u)
        for i in u:
            i=(i-m)/d
            temp.append(i)

    
    return temp



def solve(A,lamda,ifintercept=0):
    """
    lamda is the regularization parameter. lamda should be non-negetive
    ifintercept=======1 or 0.  1 to change the intercept.  0 to leave the intercept the same.

    using SVD deposition to compute psedoinverse matrix of A
    return the psedoinverse
    """
    if lamda>=0:
        u,s,v=np.linalg.svd(A) 
        regu=lamda*np.eye((np.shape(A.T)[0]))
        #print(regu)
        if not ifintercept:regu[0][0]=0
        singular=np.zeros(np.shape(A.T))
        for i in range(np.shape(s)[0]):
            singular[i][i]=s[i]/(s[i]**2+regu[i][i])
        ans=np.dot(np.dot(v.T,singular),u.T)
    else:
        raise regularizationError('Regularization Parameter Should be Non-negetive !')
      #TODO:np.ones 这里不对！！  
        
    return ans
    
def solve_2(A,lamda,ifintercept=1):
    if lamda==0: return solve(A,lamda,0)
    else:
        regu=lamda*np.eye(np.shape(A.T)[0])
        #print(regu)
        if not ifintercept:regu[0][0]=0   
        #TODO:
        return np.dot(np.linalg.inv(np.dot(A.T,A)+regu), A.T)



def showMatrix(m):
    """
    convert matrix to image
    """
    fig=plt.figure()
    plt.imshow(m,cmap='gray')
    return

def printMatrix(m):
    """
    to print each element of the matrix
    """
    [rows, cols] = np.shape(m)
    for i in range(rows - 1):
        for j in range(cols-1):
            print(m[j, i])




def threeDplot(signal,label):
    """
    to plot 3D-line-image
    """
    mpl.rcParams['legend.fontsize']=10
    fig=plt.figure()
    ax=fig.gca(projection='3d')
    ax.plot(signal[0],signal[1],signal[2],label='%s'%label)
    ax.legend()
    plt.show()
    return

def plotState(states,shownum,showmode=1):
    """
    showmode=1 : show by row. (plot different rows )
    showmode=0 : show by column (plot different columns)
    
    """
    rd=np.random.randint(0, np.shape(states)[1-showmode], size=shownum)
    rd=list(rd)
    print(rd)
    x=states[rd,:]
    print(np.shape(x))
    timelen=np.shape(x)[-1]
    fig=plt.figure()
    b=np.linspace(0,shownum,shownum)
    a=np.linspace(0,timelen,timelen)
    a,b=np.meshgrid(a,b)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(a,b,x,rstride=1,cstride=0)
    return





    


def goldenOpt(a,b,f,Theta_error,ifplot):
    """
    0.618方法
    """
    dic={}
    r=(sqrt(5)-1)/2  
    a1=b-r*(b-a)  
    print(a1)
    a2=a+r*(b-a)  
    stepNum=0  
    flag1=1
    flag2=1
    while abs(b-a)>Theta_error:  
        stepNum=stepNum+1  
        if flag1:
            f1=f(a1) 
            dic[a1]=f1
        if flag2:
            f2=f(a2)  
            dic[a2]=f2
        if f1>f2:  
            a=a1  
            f1=f2  
            a1=a2  
            a2=a+r*(b-a)  
            flag1=0
            flag2=1
        else:  
            b=a2  
            a2=a1  
            f2=f1  
            a1=b-r*(b-a) 
            flag2=0
            flag1=1
    x_opt=(a+b)/2  
    f_opt=f(x_opt)  
    x=[]
    y=[]
    for key,value in dic.items():
        x.append(key)
        y.append(value)
    if ifplot:plt.plot(x,y)
    return (x_opt,f_opt,stepNum)

def mfcc(filename,delta,framenum):
    y,sr=librosa.load(filename, sr=16000)
#    plt.figure()
#    plt.plot(y)
#    plt.title('original audio')
#    plt.xlabel('time')
#    plt.ylabel('energy')
    mfcc_feat=sf.mfcc(signal=y,samplerate=sr,winlen=0.025,nfft=512,winstep=0.01,numcep=13,ceplifter=22)   #?????appendEnergy ceplifter?? 
    if delta==0:
        return mfcc_feat
    elif delta==1:
        mfcc_delta=sf.delta(mfcc_feat,framenum) # calculate delta features based on preceding and following 2 frames
        return np.hstack((mfcc_feat,mfcc_delta))
    elif delta==2:
        mfcc_delta=sf.delta(mfcc_feat,framenum)
        mfcc_delta_2=sf.delta(mfcc_delta,framenum)
        return np.hstack((mfcc_feat,mfcc(filename,1,framenum)))
    else:
        return mfcc_feat

def get_label_file(path,filename):
    path_annot=os.path.join(path,'sources','annote')
    annotAbs_path=os.path.join(path_annot,os.path.splitext(filename)[0]+'.lab')
    return annotAbs_path


def match_label(file,winsize=0.025, step=0.01, mode=0):
    """
    """
    nodes=[]
    f=open(file,'r')
    y=[]
    s=f.readlines()
    i=1
    if mode==1:
        for line in s:
                line=line.strip()
                line=re.split(r' *',line)
                if 'no' in line[2]:
                    nodes.append((float(line[0])/0.025,0))
                    while (i-1)*step+winsize < float(line[1]):
                        y.append((i,0))
                        i+=1
                    if ((i-1)*step+winsize<float(line[1])+0.0125)  :
                        y.append((i,-1))
                        i+=1
                    
                        # del the frame which contain less than 50% singing      
                else:
                    nodes.append((float(line[0])/0.025,1))
                    while (i-1)*step+winsize <float(line[1]):
                        y.append((i,1))
                        i+=1
                    if (i-1)*step+winsize < float(line[1])+0.0125:
                        y.append((i,1))
                        i+=1
    if mode==0:
        # del the frame which is not full of vocals
        for line in s:
            line=line.strip()
            line=re.split(r' *',line)
            if 'no' in line[2]:
                nodes.append((float(line[0])/winsize,0))
                while (i-1)*step+winsize < float(line[1]):
                    y.append((i,0))
                    i+=1
                y.append((i,-1))
                i+=1
                
            else:
                nodes.append((float(line[0])/winsize,1))
                while (i-1)*step+winsize <float(line[1]):
                    y.append((i,1))
                    i+=1
                y.append((i,-1))
                i+=1
        
    label=[x[1] for x in y if x[1]!=-1]
    del_list=[x[0] for x in y if x[1]==-1]      # Notice: min of index is 1 instead of 0
    return label, del_list
 

def divide(nparray,divideNum):
    l=np.shape(nparray)[-1]
    l0=l//divideNum
    refer=[list(range(i*l0,(i+1)*l0)) for i in range(divideNum-1)]
    refer.append(list(range((divideNum-1)*l0,l)))
    
    dic=[]
    
    for i in range(divideNum):
        train=np.delete(nparray,refer[i],axis=1)
        valid=nparray[:,refer[i]]
        dic.append((train,valid))
    
    return dic
    
                
                
 
    
    

    
            
    
    
    
    
    
    
    
    