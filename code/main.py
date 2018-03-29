#%% EEL 4930 Final Project
# Christian Marin
# Troy Tharpe
# Dean Fortier

import numpy as np # (given)
import matplotlib.pyplot as plt # (given)
import scipy.io as spio # (given)

import cv2

from skimage.segmentation import felzenszwalb, slic, quickshift

RGB_FLAG = [255,0,255]

#%% general data i/o

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
            data_train_labels[y[j],x[j],:] = RGB_FLAG
    for i in range(8): # pond masks
        y = pond_masks[i][:,1]
        x = pond_masks[i][:,0]
        for j in range(np.size(pond_masks[i],axis=0)):
            data_train_masks[y[j],x[j],:] = RGB_FLAG
    cv2.imwrite('../data/data_train_original.png',data_train[:,:,::-1])
    cv2.imwrite('../data/data_train_labels.png',data_train_labels[:,:,::-1])
    cv2.imwrite('../data/data_train_masks.png',data_train_masks[:,:,::-1])


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


#%%  preprocessing
    
def segmentation_canny(data,d,sigmaColor,sigmaSpace,minVal,maxVal):
    ''' performs edge detection using canny filter
            d: diameter of pixel neighborhoods used during filtering
            sigmaColor: color space filter sigma
                (larger values, more distinct colors will blur)
            sigmaSpace: coordinate space filter sigma
                (larger value, farther pixels will blur)
            minVal: lower threshold (less significant edges are discarded)
            maxVal: higher threshold (more significant edges are preserved)
    '''
    smooth = cv2.bilateralFilter(data,d,sigmaColor,sigmaSpace)
    edges = cv2.Canny(smooth,minVal,maxVal,L2gradient=True)
    # todo - close open canny contours for better segmentation
    data[edges>0,:] = RGB_FLAG
    cv2.imwrite('../data/segmentation_canny.png',data[:,:,::-1])


def segmentation_felzenszwalb(data,scale,sigma,min_size,\
    d,sigmaColor,sigmaSpace,minVal,maxVal):
    ''' performs segmentation using felzenszwalb
            scale: higher value, larger clusters
            sigma: gaussian filter width
            min_size: min component size
            <canny params, refer to segmentation_canny()>
    '''
    segments = felzenszwalb(data,scale,sigma,min_size)
    edges = cv2.Canny(segments,minVal,maxVal,L2gradient=True)
    data[edges>0,:] = RGB_FLAG
    cv2.imwrite('../data/segmentation_felzenszwalb.png',data[:,:,::-1])


def segmentation_quickshift(data,ratio,kernel_size,max_dist,return_tree,sigma,\
    d,sigmaColor,sigmaSpace,minVal,maxVal):
    ''' performs segmentation using quickshift
            ratio: 
            kernel_size: 
            max_dist: 
            return_tree: 
            sigma: 
            <canny params, refer to segmentation_canny()>
    '''
    segments = quickshift(data,)
    edges = cv2.Canny(segments,minVal,maxVal,L2gradient=True)
    data[edges>0,:] = RGB_FLAG
    cv2.imwrite('../data/segmentation_quickshift.png',data[:,:,::-1])


#%%  main

def main():
    
    data_train,locs,labels,pond_masks = read_train_data()
    #write_train_data(data_train,locs,labels,pond_masks)
    #plot_train_labels(data_train,labels,locs)
    #plot_train_masks(np.size(data_train,axis=0),pond_masks)
    
    segmentation_canny(data_train.copy(),9,75,75,100,200)
    segmentation_felzenszwalb(data_train.copy(),3,.95,5,9,75,75,100,200)
    segmentation_quickshift(data_train.copy(),1.0,5,10,False,0,9,75,75,100,200)


if  __name__ == '__main__':
    main()

