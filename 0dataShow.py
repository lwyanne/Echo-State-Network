# -*- coding: utf-8 -*-
"""
Data Show

Created on Mon Aug 20 14:29:34 2018

@author: acerr
"""
from function import *
#%%    n_reservoir=1000   test error regarding sparsity(==sparsity_in) & spectral radius

#x,y=np.meshgrid(x,y)
#ax = fig.add_subplot(111, projection='3d')
#ax.plot_wireframe(x,y,matrix.T,rstride=1,cstride=1)
#
#
#plt.figure()
#plt.pcolormesh(x,y,matrix.T)
#plt.colorbar()
#plt.xticks(np.linspace(0,1,15))
#plt.xlabel('spectral radius')
#plt.yticks([0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01,0.012,0.015,0.1])
#plt.ylabel('sparsity')
#plt.title('reservoir_size=1000')


y = [0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01,0.012,0.015,0.1]
x = np.linspace(0,1,15)
matrix= np.load(os.path.join(path_data,'candsparsity.npy'))
matrix=matrix.T



fig, ax = plt.subplots()
im = ax.imshow(matrix)

# We want to show all ticks...
ax.set_xticks(np.arange(len(x)))
ax.set_yticks(np.arange(len(y)))
# ... and label them with the respective list entries
ax.set_xticklabels(['%.3f'%i for i in x])
ax.set_yticklabels(y)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(y)):
    for j in range(len(x)):
        text = ax.text(j, i, '%.3f'%matrix[i, j],
                       ha="center", va="center", color="w")

ax.set_title("testing error")
fig.tight_layout()
plt.show()


"""
sparsity=0.005 is the best 

>>np.unravel_index(np.argmin(matrix,axis=None),matrix.shape)
Out[39]: (6, 2)       spectral radius = 0.4285   sparsity = 0.005



minError======0.18806807193966346


np.argmin(matrix,axis=0)
Out[44]: array([10, 14,  6,  4,  2, 10, 12,  4,  2,  9, 10], dtype=int64)

np.argmin(matrix,axis=1)
Out[45]: 
array([10,  8,  8,  7,  8,  2,  2,  2,  8,  2,  2,  2,  2,  2,  8],
      dtype=int64)

"""



#%% degree=5
fig=plt.figure()
matrix= np.load(os.path.join(path_data,'c_reservoirSiz.npy'))
x=np.linspace(0,1,15)
y=[100,150,200,350,400,450,500,550,600,650,700,750,800,850,900,1000]
matrix=matrix.T



fig, ax = plt.subplots()
im = ax.imshow(matrix)

# We want to show all ticks...
ax.set_xticks(np.arange(len(x)))
ax.set_yticks(np.arange(len(y)))
# ... and label them with the respective list entries
ax.set_xticklabels(['%.3f'%i for i in x])
ax.set_yticklabels(y)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(y)):
    for j in range(len(x)):
        text = ax.text(j, i, '%.3f'%matrix[i, j],
                       ha="center", va="center", color="w")

ax.set_title("testing error")
fig.tight_layout()
plt.show()


a,b=np.unravel_index(np.argmin(matrix, axis=None), matrix.shape)
print('optimal spectral radius==%f, optimal size==%f'%(x[a],y[b]))
print('minimal error===%f'%np.min(matrix))
"""
Error==0.1933
optimal spetral radius==0.714286, optimal size==1000.000000
"""

#%% reservoir size & sparsity
fig=plt.figure()
matrix= np.load(os.path.join(path_data,'n_sparsity.npy'))
x=range(100,1100,50)
y=[0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01,0.012,0.015,0.1]
matrix=matrix.T



fig, ax = plt.subplots()
im = ax.imshow(matrix)

# We want to show all ticks...
ax.set_xticks(np.arange(len(x)))
ax.set_yticks(np.arange(len(y)))
# ... and label them with the respective list entries
ax.set_xticklabels(['%.3f'%i for i in x])
ax.set_yticklabels(y)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(y)):
    for j in range(len(x)):
        text = ax.text(j, i, '%.3f'%matrix[i, j],
                       ha="center", va="center", color="w")

ax.set_title("testing error")
fig.tight_layout()
plt.show()
#plt.title('reservoir size & sparsity')

"""
error===0.1836   N==900,  sparsity==0.004    degree=4

np.argmin(matrix,axis=0)
Out[21]: array([15, 16, 16, 18, 14, 19, 13, 14, 19, 16, 19], dtype=int64)

np.argmin(matrix,axis=1)
Out[22]: 
array([ 9, 10,  9,  9,  6,  7,  6,  1,  8,  7,  8,  6,  7,  6,  7,  0,  1,
        2,  8,  8], dtype=int64)
    
    
"""
