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




class Lorenz():
    """
    This is used to generate trajectory of Lorenz System.
    (Interate by Heun Method)
    h===========the integration step size
    len=========the length of time series 

    """
    
    def __init__(self,init_point=(1,1,1),len=200000,r=28,sigma=10,beta=8/3.0,h=0.005):
        self.length=len
        self.r=r
        self.sigma=sigma
        self.beta=beta
        self.h=h    
        self.u=np.zeros((3,len))
        self.u[:,0]=list(init_point)
        #using Heun method for numerical solution
        for i in range(1,len):
            temp0=self.u[0][i-1]+self.h*self.sigma*(self.u[1][i-1]-self.u[0][i-1])    
            temp1=self.u[1][i-1]+self.h*(self.u[0][i-1]*(self.r-self.u[2][i-1])-self.u[1][i-1])
            temp2=self.u[2][i-1]+self.h*(self.u[0][i-1]*self.u[1][i-1]-self.beta*self.u[2][i-1])
            self.u[0][i]=self.u[0][i-1]+self.h/2*(self.sigma*(self.u[1][i-1]-self.u[0][i-1])+self.sigma*(temp1-temp0))
            self.u[1][i]=self.u[1][i-1]+self.h/2*(self.u[0][i-1]*(self.r-self.u[2][i-1])-self.u[1][i-1]+temp0*(self.r-temp2)-temp1)
            self.u[2][i]=self.u[2][i-1]+self.h/2*(self.u[0][i-1]*self.u[1][i-1]-self.beta*self.u[2][i-1]+temp0*temp1-self.beta*temp2)

        #discard the Transient 10s
        for i in range(3):
            temp=self.u
            del self.u
            self.u=temp[:,2000:]
    
    def downsample(self,interval=10):
        self.interval=interval
        r=[]
        for j in range(np.shape(self.u)[0]):
            r.append([])
            for i in range(len(self.u[j])):
                if i% self.interval ==0:
                    r[j].append(self.u[j][i]) 
        self.u=r

    def normalize(self):
        for i in range(3):
            self.u[i]=normal(self.u[i],1)


    def show(self):
        mpl.rcParams['legend.fontsize']=10
        fig=plt.figure()
        ax=fig.gca(projection='3d')
        ax.plot(self.u[0],self.u[1],self.u[2])
        ax.legend()
        plt.title('Lorenz time series , each step is %f'%(self.h*self.interval))
        plt.show()
    
    def get(self):
        return self.u

        

class ESN():

    def __init__(self,n_inputs,n_outputs,
                 n_reservoir=200,
                spectral_radius=0.95,
                sparsity=0.1, 
                sparsity_in=0.1,
                ifplot=0,
                noise=0.001,
                seednum=42,
                b=1,
                a=1,
                alpha=0):
        """
        Args:
            n_inputs: the number of input dimensions
            n_outputs: the number of output dimensions
            n_reservoir:nr of reservoir neurons
            spectral_radius: spectral radius of the recurrent weight matrix
        """

        self.n_inputs=n_inputs
        self.n_reservoir=n_reservoir
        self.n_outputs=n_outputs
        self.spectral_radius=spectral_radius
        self.sparsity=sparsity
        self.sparsity_in=sparsity_in
        self.noise=noise
        self.flag=self.n_inputs-1
        self.seednum=seednum
        self.b=b
        self.a=a
        self.ifplot=ifplot
        self.alpha=alpha
    
    def initweights(self):
        """
        initialize recurrent weights:
        """
        np.random.seed(self.seednum)

        W=np.random.normal(size=(self.n_reservoir,self.n_reservoir))

        n_zero = round(self.n_reservoir*self.sparsity) # the number of zero elements in each

        for i in range(self.n_reservoir):
            self.dic=list(map(
                round,(self.n_reservoir*np.random.rand(n_zero))))
            for j in range(self.n_reservoir):
                if j not in self.dic:
                    W[i,j]=0
            del self.dic

        radius=np.max(np.abs(scipy.linalg.eigvals(W)))
        self.W=W*self.spectral_radius/radius
        del n_zero

        np.random.seed(self.seednum+10)
        self.nonzero=int(round(self.n_reservoir*self.sparsity_in))
        #initialize input weights:
        np.random.seed(self.seednum+1)
        V=np.random.normal(size=(self.n_reservoir,self.n_inputs))


        for i in range(self.n_inputs):
            np.random.seed(self.seednum+2+i)        
            dic=list(map(round,(self.n_reservoir*np.random.rand(self.nonzero))))
            for j in range(self.n_reservoir):
                if j not in dic:
                    V[j,i]=0
                    
        del dic
        self.V=V
        #initialize bias weights:
        np.random.seed(self.seednum+4)        
        self.Wb=self.b*np.random.normal(size=(self.n_reservoir,1))
        np.random.seed(self.seednum+5)
        self.noiseVec=self.noise * np.random.normal(size=(self.n_reservoir,1)).T    
    
    def investigate(self,n):
        np.random.seed(self.seednum+2+n)        
        dic=list(map(round,(self.n_reservoir*np.random.rand(self.nonzero))))    
        plt.figure()
        plt.subplot(221)
        plt.plot(self.inputs[n])
        plt.title('the Nth feature')
        plt.subplot(222)
        print(dic[1])
        plt.plot(self.state[int(dic[1])])
        plt.title('internal node trajectory')
        plt.subplot(223)
        plt.plot(self.state[int(dic[2])])
        plt.title('internal node trajectory')
        plt.subplot(224)
        plt.plot(self.state[int(dic[3])])
        plt.title('internal node trajectory')
        
    
    def update(self,inputs,ifrestart=1):
        """
        update the state of internal nodes.
        """
        self.lenth=int(np.size(inputs)/self.n_inputs)
        inputs=np.reshape(inputs, (self.n_inputs,self.lenth))  
        self.inputs=inputs

        if ifrestart:
            self.state=np.zeros((self.n_reservoir,self.lenth))
            self.state[:,0]=np.dot(self.V,inputs[:,0].T)

        else:
            self.state=np.hstack(
                (self.laststate,
                np.zeros((self.n_reservoir,self.lenth)))
                )      

            inputs=np.hstack((self.lastinput,inputs))

            self.lenth+=1       

        for i in range(1, self.lenth):
            self.state[:,i]=(
                self.alpha * self.state[:,i]
                +
                (1 - self.alpha) * np.tanh(
                    np.dot(self.W.T, self.state[:,i-1])
                        + self.a * np.dot(self.V, inputs[:,i].T) 
                        + self.Wb.T
                    ) 
                    + self.noiseVec  #TODO:
                    #+ self.noise * np.random.normal(size=(self.n_reservoir,1)).T   
            )       



        self.laststate=self.state[:,-1]
        self.laststate=np.reshape(self.laststate, (len(self.laststate),-1))
        self.lastinput=inputs[:,-1]
        self.lastinput=np.reshape(self.lastinput, (len(self.lastinput),-1))
        self.bias=np.ones((1,self.lenth))
        self.allstate=np.vstack((self.bias,self.state))
        if not ifrestart:
            temp=np.delete(self.allstate,0,1)
            del self.allstate
            self.allstate=temp
            del temp       

    def show_internal(self, shownum=5,ifshow=1):
        """show the internal activation trajectory
        :param ifshow: 'plt.show()'
        """
        plotState(self.allstate,shownum)
        if ifshow: plt.show()
        
        
        

    def fit(self, inputs,targets,namda,ifintercept=1):
        """
        fit the output weights, using Ridge Regression methods.
        return: the coefficience matrix
        NOTICE: namda should be non-negative
        """   
        targets=np.array(targets) 
        inputs=np.array(inputs)       
        if inputs.ndim < 2:
            inputs = np.reshape(inputs, (-1,len(inputs)))
        if targets.ndim < 2:
            targets = np.reshape(targets, (-1,len(targets)))
        # self.update(inputs,ifrestart) 
        #self.state=discard(self.state)
        #targets=discard(targets)
        #self.bias=np.ones((1,np.shape(targets)[-1]))
        #self.allstate=np.vstack((self.bias,self.state))
        if namda==0: 
            self.coefs=np.dot(
                    np.dot(
                    np.linalg.inv(
                            np.dot(self.allstate,self.allstate.T)
                    ),self.allstate
                    ),targets.T
                    )
            return
        self.coefs=np.dot(solve_2(self.allstate.T,namda,ifintercept),targets.T)
    
    
        
    def ridgeRegres(self,u_train,u_target,u_valid,u_true,a=-8,b=3,ifdiscard=0,ifload=0,ifplot=0):
        """ Ridge Regression
        
        :param u_train:
        """
        #plt.figure()
        #plt.title('timeshift=%d'%(timeshift))
        #plt.plot(u_true,label='true')
        print('n_reservoir===',self.n_reservoir)
        if self.n_inputs==1:
            len1=len(u_train)
            len2=len(u_valid)
            u_train=np.reshape(np.array(u_train),(-1,len1))
            u_target=np.reshape(np.array(u_target),(-1,len1))
            u_valid=np.reshape(np.array(u_valid),(-1,len2))
            u_true=np.reshape(np.array(u_true),(-1,len2))
            error=[]
        error_train=[]
        para=0
        if not ifload:
            print('n_reservoir===',self.n_reservoir)
            self.update(u_train[:,:64],1)
            print('shape allstate========',np.shape(self.allstate))
            
            del self.allstate
            print('..\n initially updated and discarded the transient of 64 steps\n...')
            self.update(u_train,0)
            print('..\nupdated training inputs')
            #self.show_internal(shownum=3)
            #self.investigate(1)
            global temp1, temp2
            temp1=self.allstate
            if ifdiscard:temp1=discard(temp1)
    
            self.update(u_valid,0) 
            #self.show_internal(shownum=3)
            #self.investigate(1)
            print('...\nupdated valid inputs')
            temp2=self.allstate
            if ifdiscard:temp2=discard(temp2)
            np.save(os.path.join(path_data,'temp1'),temp1)
            np.save(os.path.join(path_data,'temp2'),temp2)
        else:
            temp1=np.load(os.path.join(path_data,'temp1.npy'))
            temp2=np.load(os.path.join(path_data,'temp2.npy'))       
        
        def tune(lamda):
            y=10**lamda
            self.allstate=temp1
            self.fit(u_train,u_target,y,1)
            self.predict()
            err1=self.err(self.outputs,u_target,1)
            print('****training====',err1)
 
            self.allstate=temp2
            self.predict()
            err2=self.err(self.outputs,u_true,1)
            print('*****valid===',err2)
            #print('lamda===',lamda,'error===',err2)
            return err2 
        


        
        theta_error=0.1
        x_opt,f_opt,stepnum=goldenOpt(a,b,tune,theta_error,ifplot)
        err2=self.err(self.outputs,u_true,-1)
        print('testing classification error==',err2)
        
        self.allstate=temp1
        print('shape allstate========',np.shape(self.allstate))
        self.predict()
        err1=self.err(self.outputs,u_target,-1)
        err11=self.err(self.outputs,u_target,1)
        print('training classification error==',err1)
        print('training NMSE==',err11)        
        plt.figure()
        plt.plot(self.outputs[0],'.',color='brown',markersize=1.2,label='training output')
        plt.plot(u_target[0],color='dimgray',linewidth=0.8,label='label')
        plt.title('training')
        plt.legend()
#        
        self.allstate=temp2
        self.predict()       
        plt.figure()   
        plt.plot(self.outputs[0],'.',color='brown',markersize=1.2,label='test output')
        plt.plot(u_true[0],color='dimgray',linewidth=0.8,label='label')
        plt.legend()
#       
        print('lambda==',10**x_opt,'minimum testing NMSE==', f_opt,
              'stepnum==',stepnum)
        return f_opt,x_opt
        


    def predict(self):

        self.outputs=np.dot(self.allstate.T,self.coefs).T



    

    def err(self,signal,real,mode):
        """
        calculate the error.
        mode===1 :  use Normalized Mean Square Error
        mode===0:   use Mean Square Error
        mode==-1:   classification error
        """
        # get the length
        self.siglenth=int(np.size(signal)/self.n_outputs)
        # reshape the two signals in case one is column vector, 
        # and the other is row vector
        if self.n_outputs==1:
            real=np.reshape(np.array(real),(1,self.siglenth))
            signal=np.reshape(np.array(signal),(1,self.siglenth))     
        if mode==1:
            err=(np.mean(
                    np.multiply(
                            (signal-real),(signal-real)
                            ))/
                    np.mean(np.multiply(
                            real-np.mean(real),real-np.mean(real)
                            )
            ))
        
        elif mode==0:
            err=(np.sum(np.multiply((signal-real),(signal-real))
        ))/self.siglenth
        
        else:
            right=0
            for i in range(self.siglenth):
                if signal[0,i]>=0.5 and real[0,i]==1:
                    right+=1   
                elif signal[0,i]<0.5 and real[0,i]==0:
                    right+=1
            err=1-(right/self.siglenth)                                                                                                                                        
        return err
  




    def mydel(self,mode):
        del self.allstate
        del self.state
        del self.bias

        if mode: del self.coefs
        
    


class ESN_complex(ESN):
    pass


class LESN(ESN):
    def update(self,inputs,ifrestart):
        """
        update the state of internal nodes.
        """
        self.lenth=int(np.size(inputs)/self.n_inputs)
        inputs=np.reshape(inputs, (self.n_inputs,self.lenth))  

        if ifrestart:
            self.state=np.zeros((self.n_reservoir,self.lenth))
            self.state[:,0]=np.dot(self.V,inputs[:,0].T)

        else:
            self.state=np.hstack(
                (self.laststate,
                np.zeros((self.n_reservoir,self.lenth)))
                )      

            inputs=np.hstack((self.lastinput,inputs))

            self.lenth+=1       

        for i in range(1,self.lenth):
            self.state[:,i]=(
                    np.dot(self.W.T, self.state[:,i-1])
                        + self.a * np.dot(self.V, inputs[:,i].T) 
                        + self.Wb.T
                    + self.noiseVec  #TODO:
                    #+ self.noise * np.random.normal(size=(self.n_reservoir,1)).T   
            )       

        self.laststate=self.state[:,-1]
        self.laststate=np.reshape(self.laststate, (len(self.laststate),-1))
        
        self.lastinput=inputs[:,-1]
        self.lastinput=np.reshape(self.lastinput, (len(self.lastinput),-1))
        self.bias=np.ones((1,self.lenth))
        self.allstate=np.vstack((self.bias,self.state))
        if not ifrestart:
            temp=np.delete(self.allstate,0,1)
            del self.allstate
            self.allstate=temp
            del temp      

class trivial(ESN):
    pass


