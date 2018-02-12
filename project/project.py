
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spio

# Load Data
mat = spio.loadmat('data_train.mat')
data_train = mat.get('data_train')
N = np.size(data_train,axis=0)

# Load Object Locations and Labels
indices = np.loadtxt('labels.txt')
locations = indices[:,0:2]
labels = indices[:,2]
# 1 - White Cars, 2 - Red Cars, 3 - Pools, 4 - Ponds
no_labels = np.unique(labels)

# Load Pond Mask
pond1 = np.loadtxt('pond1.txt')
pond2 = np.loadtxt('pond2.txt')
pond3 = np.loadtxt('pond3.txt')
pond4 = np.loadtxt('pond4.txt')
pond5 = np.loadtxt('pond5.txt')
pond6 = np.loadtxt('pond6.txt')
pond7 = np.loadtxt('pond7.txt')
pond8 = np.loadtxt('pond8.txt')

# Plot Training Image with Objects center-point marked
fig = plt.figure()
colors = ['w','r','b','g']
ll = []
plt.imshow(data_train)
for i in range(len(no_labels)):
    lbl = plt.scatter(locations[labels==(i+1),0],locations[labels==(i+1),1],c=colors[i])
    ll = np.append(ll,lbl)
plt.legend(ll,['White Cars','Red Cars','Pools','Ponds'])
plt.title('Training Data')
plt.show()

#%%
# Create a mask from locations 
def Pond_Mask(N,M,locations):
    Mask = np.zeros((N,M))
    locations=locations.astype(int)
    Mask[locations[:,1],locations[:,0]] = 1
    return Mask

# Plot masks for all ponds
Ponds = np.zeros(((N,N,8)))
for i in range(8):
    fig = plt.figure()
    Ponds[:,:,i] = Pond_Mask(N,N,eval('pond'+str(i+1)))
    plt.imshow(Ponds[:,:,i])
    plt.title('Pond '+str(i+1))
    plt.show()




