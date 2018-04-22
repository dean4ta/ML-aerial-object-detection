#%% EEL 4930 Final Project
# Christian Marin
# Troy Tharpe
# Dean Fortier

import numpy as np # (given)
import matplotlib.pyplot as plt # (given)
import scipy.io as spio # (given)

import cv2

from roll import rolling_window

#%% load data

def load_train_data():
    ''' loads provided training data from 'data' folder in root directory
            data_train: (6250,6250,3) (ypixel,xpixel,r/g/b)
            locs: (112,2) (index,xpixel/ypixel)
            labels: (112,1) (index,label)
            pond_masks: (?,2) (index,xpixel/ypixel)
        ex) data_train,locs,labels,pond_masks = read_train_data()
    '''
    # load data_train.mat: format - <y>,<x>,<r,g,b>
    mat = spio.loadmat('../data/data_train.mat') # load mat object
    data_train = mat.get('data_train') # extract image struct from mat object
    # load labels.txt: format - <x> <y> <label>
    pairs = np.loadtxt('../data/labels.txt').astype(np.uint16) # load label matrix
    locs = pairs[:,0:2] # <x> <y> pixel indices
    labels = pairs[:,2].astype(np.uint8) # <label> 1=whitecar,2=redcar,3=pool,4=pond
    # load pond#.txt masks: format - <x> <y>
    pond_masks = []
    for i in range(8):
        file = '../data/pond'+str(i+1)+'.txt'
        pond_masks.append(np.loadtxt(file).astype(np.uint16))
    return data_train,locs,labels,pond_masks

#%% preprocessing

def get_canny(data,d=9,σColor=75,σSpace=75,minVal=100,maxVal=200):
    ''' performs edge detection using canny filter
            d: diameter of pixel neighborhoods used during filtering
            σColor: color space filter σ (larger -> more distinct colors will blur)
            σSpace: coord space filter σ (larger -> more distant pixels will blur)
            minVal: low threshold (less significant edges are discarded)
            maxVal: high threshold (more significant edges are preserved)
    '''
    return cv2.Canny(cv2.bilateralFilter(data,d,σColor,σSpace),minVal,maxVal,L2gradient=True)

def get_fourier(data,HPF_size=60):
    ''' Perofrms a DFT and high pass filtering
        data: grayscale 2D image array
        HPF_size: High Pass Filter size of box to filter out
    '''
    r,c,d = np.shape(data)
    rcenter,ccenter = int(r/2),int(c/2)
    # data = cv2.fastNlMeansDenoising(data,None,10,7,21)
    
    
    mask = np.ones((r,c,2),np.uint8)
    mask[rcenter-HPF_size:rcenter+HPF_size,ccenter-HPF_size:ccenter+HPF_size] = 0
    fshift = dft_shift*mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
    
    dft = cv2.dft(np.float32(data),flags=cv2.DFT_COMPLEX_OUTPUT)
    dft = np.fft.fftshift(dft)
    dft[rcenter-HPF_size:rcenter+HPF_size,ccenter-HPF_size:ccenter+HPF_size] = 0
    return cv2.idft(np.fft.ifftshift(dft))
    '''
    data = (data/np.max(data)*255)**1.7
    data[np.where(data>255)] = 255
    return data'''

#%% feature extraction

def extract_features(data,win_y,win_x):
    ''' extract features from given image over sliding window
            data - 3D image (2D image x color dimensions)
            win_y - window height SHOULD BE ODD
            win_x - window width  SHOULD BE ODD
    '''
    N1,N2,C = np.shape(data)
    '''
    features = np.zeros((N1-win_y+1,N2-win_x+1,C*nfeatures))
    feature_iter = 0
    for x in range(C):
        print(' computing sliding windows for',x,'color dimension')
        windows = rolling_window(data[:,:,x],(win_y,win_x))
        # ***** modify below - add features as desired ***** #
        # features[:,:,feature_iter+n] = f(windows,axis=(2,3)) #
        print(' computing features for',x,'color dimension')
        # features[:,:,feature_iter+0] = np.mean(windows,axis=(2,3)) -- update to be self pixel
        features[:,:,feature_iter+0] = np.mean(windows,axis=(2,3))
        features[:,:,feature_iter+1] = np.median(windows,axis=(2,3))
        feature_iter += nfeatures
    '''
    color_hist = cv2.calcHist(windows,np.arange(C),None,[8],[0,255])
    
    return features

#%%  main

def main():
    
    data,locs,labels,pond_masks = load_train_data()
    
    hsv = cv2.cvtColor(data,cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(data,cv2.COLOR_RGB2GRAY)[:,:,None]
    canny = get_canny(data)[:,:,None]
    fourier = get_fourier(data)
    data = np.concatenate((data,hsv,gray,canny,fourier),axis=2)
    hsv,gray,canny,fourier = 0,0,0,0
    np.save('../data/data_preproc.mat',data)
    data = np.load('../data/data_preproc.mat')
    
    extract_features(data,)
    # np.save('../data/data_features.mat',features)
    # features = np.load('../data/data_features.mat')
    
    
    # dft = cv_dft('../data/data_train_original.png',a=1,b=2) # dft w/o denoising
    # denoise_colored_image('data_train_denoised.png')
    # denoised_dft = cv_dft('data_train_denoised.png',a=3,b=4) # #dft w denoising
    # cv2.bilateralFilter('../data/data_train_denoised.png',9,75,75)
    # denoised_dft = cv_dft('../data/data_train_denoised.png',a=3,b=4) #dft w bilateral
    # data_dft = cv_dft(data_gray,a=1,b=2) # Get FT preprocess data
    # denoised_data = np.arrya(denoise_colored_image(data_train)).astype(np.uint8)
    # denoised_data_dft = np.array(cv_dft(denoised_data,a=3,b=4)).astype(np.uint8)[:,:,None]

    ## feature extraction
    # features = extract_features(data_train,6,6,2)
    # features = extract_features(colorData,win_y,win_x,nfeatures) # Extract Features from Color Space

if  __name__ == '__main__':
    main()
