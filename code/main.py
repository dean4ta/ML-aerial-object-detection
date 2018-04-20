#%% EEL 4930 Final Project
# Christian Marin
# Troy Tharpe
# Dean Fortier

import numpy as np # (given)
import matplotlib.pyplot as plt # (given)
import scipy.io as spio # (given)
import copy

import cv2

from numba import jit
from roll import rolling_window
from sklearn.metrics import f1_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#%% data i/o

def read_train_data():
    ''' loads provided training data from 'data' folder in root directory
            data_train: (6250,6250,3) (ypixel,xpixel,r/g/b)
            locs: (112,2) (index,xpixel/ypixel)
            labels: (112,1) (index,label)
            pond_masks: (?,2) (index,xpixel/ypixel)
    '''
    # load data_train.mat: format - <y>,<x>,<r,g,b>
    mat = spio.loadmat('../data/data_train.mat') # load mat object
    data_train = mat.get('data_train') # extract image struct from mat object
    # load labels.txt: format - <x> <y> <label>
    pairs = np.loadtxt('../data/labels.txt').astype(int) # load label matrix
    locs = pairs[:,0:2] # <x> <y> pixel indices
    labels = pairs[:,2] # <label> 1=whitecar,2=redcar,3=pool,4=pond
    # load pond#.txt masks: format - <x> <y>
    pond_masks = []
    for i in range(8):
        file = '../data/pond'+str(i+1)+'.txt'
        pond_masks.append(np.loadtxt(file).astype(int))
    return data_train,locs,labels,pond_masks


def write_train_data(data_train,locs,labels,pond_masks):
    ''' write original and labeled image as png
            data_train_original.png: original matrix as png
            data_train_labeled.png: pixel-level target label mask (0xFF00FF)
    '''
    data_train_labels = data_train.copy() # copy training image
    data_train_masks = data_train.copy() # copy training image
    for i in range(len(np.unique(labels))): # unique pixel labels
        y = np.array(locs[labels==(i+1),1],dtype=int)
        x = np.array(locs[labels==(i+1),0],dtype=int)
        for j in range(np.size(locs[labels==(i+1)],axis=0)):
            data_train_labels[y[j],x[j],:] = [255,0,255]
    for i in range(8): # pond masks
        y = pond_masks[i][:,1]
        x = pond_masks[i][:,0]
        for j in range(np.size(pond_masks[i],axis=0)):
            data_train_masks[y[j],x[j],:] = [255,0,255]
    cv2.imwrite('../data/data_train_original.png',data_train[:,:,::-1])
    cv2.imwrite('../data/data_train_labels.png',data_train_labels[:,:,::-1])
    cv2.imwrite('../data/data_train_masks.png',data_train_masks[:,:,::-1])
    
def write_pred_data(data_train,labels):
    ''' write labeled image as png
            data_pred_labeled.png: pixel-level target label mask (0xFF00FF)
    '''
    data_pred_labels = data_train.copy() # copy training image
    N1, N2 = np.shape(labels)
    for i in range(N1):
        for j in range(N2):
            if labels[i,j] != -1:
               data_pred_labels[i,j,:] = [255,0,255] 
              
               
    cv2.imwrite('../data/data_pred_labels.png',data_pred_labels[:,:,::-1])
    

def write_train_data_pixels(data_train,locs,labels,pond_masks):
    ''' write original and labeled image as png
            data_train_original.png: original matrix as png
            data_train_labeled.png: pixel-level target label mask (0xFF00FF)
    '''
    N1, N2, C = np.shape(data_train)
    data_train_labels = data_train.copy() # copy training image
    '''
        Background class label = -1
    '''
    pixel_class_labels = np.array(np.ones((N1, N2)))
    pixel_class_labels *= -1

    
    for i in range(len(np.unique(labels))): # unique pixel labels
        y = np.array(locs[labels==(i+1),1],dtype=int)
        x = np.array(locs[labels==(i+1),0],dtype=int)
        for j in range(np.size(locs[labels==(i+1)],axis=0)):
            for x_halo in range(-4,4):
                for y_halo in range(-4,4):
                    data_train_labels[y[j] + y_halo, x[j] + x_halo,:] = [255,0,255]
                    pixel_class_labels[y[j] + y_halo, x[j] + x_halo] = i +1
                        
    for i in range(8): # pond masks
        y = pond_masks[i][:,1]
        x = pond_masks[i][:,0]
        for j in range(np.size(pond_masks[i],axis=0)):
            data_train_labels[y[j],x[j],:] = [255,0,255]
            pixel_class_labels[y[j],x[j]] = 4
    cv2.imwrite('../data/data_train_original.png',data_train[:,:,::-1])
    cv2.imwrite('../data/data_train_labels.png',data_train_labels[:,:,::-1])
    
    return pixel_class_labels


#%% viualization

def plot_train_labels(data_train,labels,locs):
    ''' scatterplot target labels over training image (given, modified)
    '''
    colors,ll = ['w','r','b','g'],[] # label colors
    plt.figure() # create figure object
    plt.imshow(data_train) # add training image (background)
    for i in range(len(np.unique(labels))): # add labels (scatter plotted)
        x = locs[labels==(i+1),0]
        y = locs[labels==(i+1),1]
        lbl = plt.scatter(x,y,c=colors[i])
        ll = np.append(ll,lbl)
    plt.legend(ll,['White Car','Red Car','Pool','Pond'])
    plt.title('Training Data')
    plt.show()


def plot_train_masks(N,pond_masks):
    ''' generate and plot pond masks over empty figures (given, modified)
    '''
    decoded_masks = np.zeros((N,N,8+1)) # 0=all,1-8=standard ponds
    for i in range(8): # for every pond
        x = pond_masks[i][:,0]
        y = pond_masks[i][:,1]
        for j in range(np.size(pond_masks[i],axis=0)): # for every pixel label
            decoded_masks[y[j],x[j],0] = 1 # mark aggregate (0)
            decoded_masks[y[j],x[j],i+1] = 1 # mark individual (1-8)
        plt.title('Pond '+str(i+1))
        plt.imshow(decoded_masks[:,:,i])
        plt.show()
    plt.title('Ponds (All)')
    plt.imshow(decoded_masks[:,:,0])
    plt.show()


#%% feature extraction
@jit
def extract_features(data,win_y,win_x,nfeatures):
    ''' extract features from given image over sliding window
            data - 3D image (2D image x color dimensions)
            win_y - window height
            win_x - window width
    '''
    print('Running Feature Extractor...')
    N1,N2,C = np.shape(data)
    features = np.zeros((N1-win_y+1,N2-win_x+1,C*nfeatures))
    feature_iter = 0
    for x in range(C):
        print(' computing sliding windows for',x,'color dimension')
        windows = rolling_window(data[:,:,x],(win_y,win_x))
        # ***** modify below - add features as desired ***** #
        # features[:,:,feature_iter+n] = f(windows,axis=(2,3)) #
        print(' computing features for',x,'color dimension')
        features[:,:,feature_iter+0] = np.mean(windows,axis=(2,3))
        features[:,:,feature_iter+1] = np.median(windows,axis=(2,3))
        feature_iter += nfeatures
    print(' padding feature output to match image input dimension')  
    features = np.pad(features,((int(win_x/2) ,int(win_x/2)),(int(win_y/2) ,int(win_y/2)),(int(0), int(0))),mode = 'edge')
        
    return features

#%% Split into validation and training sets
    #splits features and lables passed in as NxN matricies
    #returns validation and training matricies as N*N x 1 array with random order

def validation_split(features, lables, valPercent):
    fN1,fN2,fC = np.shape(features)
    lN1,lN2 = np.shape(lables)
    
    #Data and lables must have same shape
    assert fN1 == lN1
    assert fN2 == lN2
        
    #place features and lables into flat arrays
    features = np.reshape(features,(fN1*fN2,fC))
    lables = lables.flatten()
    
    #shuffle samples preserving lables
    order = list(range(np.shape(features)[0]))
    np.random.shuffle(order)
    features = features[order,:]
    lables = lables[order]
    
    #split data according to valPercent
    features_train = features[0:int(fN1*fN2*(1-valPercent))]
    features_val = features[int(fN1*fN2*(1-valPercent)) + 1:(fN1*fN2)]
    lables_train = lables[0:int(fN1*fN2*(1-valPercent))]
    lables_val = lables[int(fN1*fN2*(1-valPercent)) + 1:(fN1*fN2)]
    
    return features_val,features_train, lables_val, lables_train


#%%  main

def main():
    
    data_train,locs,labels,pond_masks = read_train_data()
    pixel_class_labels = write_train_data_pixels(data_train,locs,labels,pond_masks)
    features = extract_features(data_train,5,5,2)
    
    whitePix = np.where(pixel_class_labels == 1)
    redPix = np.where(pixel_class_labels == 2)
    poolPix = np.where(pixel_class_labels == 3)
    pondPix = np.where(pixel_class_labels == 4)
    bgPix = np.where(pixel_class_labels == -1)
    bgPixCount = np.size(bgPix)/2
    
    #Split into training and validation
    features_val, features_train, lables_val, lables_train = validation_split(features, pixel_class_labels, 0)
    
    #Remove 99.9% of background data for training
    count = 0;
    mask = np.ones(len(lables_train), dtype = bool)
    for i in range(len(lables_train)):
        if lables_train[i] == -1 and count < (0.995 * bgPixCount):
            mask[i] = False
            count +=1
    nobg_lables_train = lables_train[mask,...]
    nobg_features_train = features_train[mask,...]
      
    #Perform LDA fitting
    lda = LinearDiscriminantAnalysis()
    lda.fit(nobg_features_train,nobg_lables_train)
    
    lda.fit(features_train,lables_train)
    '''
    #Remove all of background data for testing
    count = 0;
    mask = np.ones(len(lables_train), dtype = bool)
    for i in range(len(lables_train)):
        if lables_train[i] == -1:
            mask[i] = False
            count +=1
    nobg_lables_test = lables_train[mask,...]
    nobg_features_test = features_train[mask,...]
    '''
    
    #Predict lables or probability
    predicted_lables = lda.predict(features_train)
    
    predicted_lables = np.reshape(predicted_lables,(int(predicted_lables.size**(1/2)),int(predicted_lables.size**(1/2))))
    write_pred_data(data_train,predicted_lables)       
    
    #Score
    f1score = f1_score(nobg_lables_train,predicted_lables, average = 'micro')
    #f1score = f1_score(nobg_lables_test,predicted_lables, average = 'micro')
    print('f1 score = ' + f1score)
    
    #Problem!!! Too many back ground targets
    # need to lable more targets
    # cut out background targets for LDA training 
    # leave is some bakcground targets to avoid high FP Rate
    # use np.where
    
    
    


if  __name__ == '__main__':
    main()

