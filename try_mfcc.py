import os
import python_speech_features as sf
import librosa
import librosa.display
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
import numpy as np
from function import *


#%% get all the .ogg audio files under audio_train

path=os.path.abspath('.')
path_train=os.path.join(path,'sources','audio_train')
audio_train=[x for x in os.listdir(path_train) if os.path.isfile(os.path.join(path_train,x)) and os.path.splitext(x)[1]=='.ogg']
path_data=os.path.join(path,'data')


teacher=[]
flag=0


for file in audio_train:
    print("dealing with %s" %file)
    current_sample=mfcc(os.path.join(path_train,file),1)
    print(".")
    current_teacher,dellist=match_label(get_label_file(path,file))
    print("..")
    dellist.sort(reverse=True)
    dellist=[x-1 for x in dellist]
    current_samples=np.delete(current_sample,dellist,axis=0)
    if len(current_samples)-len(current_teacher)!=0:
        for i in range(len(current_samples)-len(current_teacher)):
            current_teacher.append(0)
    teacher=teacher+current_teacher
    if not flag: 
        samples=current_samples
        flag+=1
    else:samples=np.vstack((samples, current_samples))
    del current_teacher,current_sample,current_samples
    print("successfully deal with %s"% file)
    
    
teacher=np.array(teacher)
teacher=np.reshape(teacher,(1,np.size(teacher)))
samples=samples.T
np.save(os.path.join(path_data,'40ms_samples_51_delta'),samples)
np.save(os.path.join(path_data,'40ms_teacher_51_delta'),teacher)


    












#%% show spectrum


#Q: dct之后的第一个系数是干啥的，为啥很奇怪而且不要它。   appendEnergy?

path=os.path.abspath('.')
path_data=os.path.join(path,'data')
samples=np.load(os.path.join(path_data,'samples.npy'))
teacher=np.load(os.path.join(path_data,'teacher.npy'))




fig = plt.figure()
gridspec.GridSpec(8,8)

plt.subplot2grid((8,8),(0,0),7,8)
librosa.display.specshow(samples)
plt.title('MFCC')
plt.ylabel('frequecy and Delta')
plt.colorbar()
plt.subplot2grid((8,8),(7,0),1,8)
librosa.display.specshow(teacher)
plt.xlabel('time')
plt.colorbar()

#f.savefig('grid_figure.pdf')