#%% EEL 4930 Final Project
# Christian Marin
# Troy Tharpe
# Dean Fortier

import numpy as np # (given)
import matplotlib.pyplot as plt # (given)
import scipy.io as spio # (given)
import cv2,roll,util
from numba import jit
from sklearn.metrics import f1_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm

#%% load data

@jit
def load_train_data():
    ''' loads provided training data from 'data' folder in root directory
            data_train: (6250,6250,3) (ypixel,xpixel,r/g/b)
            locs: (112,2) (index,xpixel/ypixel)
            labels: (112,1) (index,label)
            pond_masks: (?,2) (index,xpixel/ypixel)
        ex) data_train,locs,labels,pond_masks = read_train_data()
    '''
    # load data_train.mat: format-<y>,<x>,<r,g,b>
    mat = spio.loadmat('../data/data_train.mat') # load mat object
    data_train = mat.get('data_train') # extract image struct from mat object
    # load labels.txt: format-<x> <y> <label>
    pairs = np.loadtxt('../data/custom_labels.txt').astype(np.uint16) # load label matrix
    locs = pairs[:,0:2] # <x> <y> pixel indices
    labels = pairs[:,2].astype(np.uint8) # <label> 1=whitecar,2=redcar,3=pool,4=pond
    # load pond .txt masks: format-<x> <y>
    pond_masks = []
    for i in range(8):
        file = '../data/pond'+str(i+1)+'.txt'
        pond_masks.append(np.loadtxt(file).astype(np.uint16))
    util.plot_train_labels(data_train,labels,locs)
    util.plot_train_masks(data_train.shape[0],pond_masks)
    return data_train,locs,labels,pond_masks

@jit
def load_custom_labels(path,dims):
    ''' load custom training labels (formatted txt from given path)
            path-string to txt file
            dims-shape of features
    '''
    N1,N2,D = dims
    labels_mask = -1*np.ones((N1,N2,1)).astype(np.uint8)
    labels = np.loadtxt(path).astype(np.uint16)
    locs,labels = labels[:,0:2],labels[:,2].astype(np.uint8)
    for i in range(labels.shape[0]):
        labels_mask[locs[i,1],locs[i,0]] = labels[i]
    return labels_mask

#%% preprocessing

@jit
def get_canny(data,d=9,σColor=75,σSpace=75,minVal=100,maxVal=200):
    ''' performs edge detection using canny filter
            d: diameter of pixel neighborhoods used during filtering
            σColor: color space filter σ (larger -> more distinct colors will blur)
            σSpace: coord space filter σ (larger -> more distant pixels will blur)
            minVal: low threshold (less significant edges are discarded)
            maxVal: high threshold (more significant edges are preserved)
    '''
    return cv2.Canny(cv2.bilateralFilter(data,d,σColor,σSpace),minVal,maxVal,L2gradient=True)

@jit
def get_fourier(data,HPF_size=60):
    ''' Perofrms a DFT and high pass filtering
        data: grayscale 2D image array
        HPF_size: High Pass Filter size of box to filter out
    '''
    r,c = int(data.shape[0]/2),int(data.shape[1]/2)
    #data = cv2.fastNlMeansDenoising(data,None,10,7,21)
    data = np.fft.fftshift(cv2.dft(np.float32(data),flags=cv2.DFT_COMPLEX_OUTPUT))
    data[r-HPF_size:r+HPF_size,c-HPF_size:c+HPF_size] = 0
    data = cv2.idft(np.fft.ifftshift(data))
    data = (data/np.max(data)*255)**2
    data[np.where(data>255)] = 255
    return (data).astype(np.uint8)

#%% feature extraction

@jit
def extract_features(data,win_y=15,win_x=15):
    N1,N2,D = np.shape(data)
    features = np.zeros((N1,N2,(8+1)*D)).astype(np.uint8) # 8=n_bins +1 for center
    for i in range(N1-win_y):
        print('iter '+str(i))
        windows = roll.rolling_window(data[i:win_y+i,:,0],(win_y,win_x)).astype(np.uint16)
        windows = np.concatenate((windows,1*255+roll.rolling_window(data[i:win_y+i,:,1],(win_y,win_x))),axis=3)
        windows = np.concatenate((windows,2*255+roll.rolling_window(data[i:win_y+i,:,2],(win_y,win_x))),axis=3)
        windows = np.concatenate((windows,3*255+roll.rolling_window(data[i:win_y+i,:,3],(win_y,win_x))),axis=3)
        windows = np.concatenate((windows,4*255+roll.rolling_window(data[i:win_y+i,:,4],(win_y,win_x))),axis=3)
        windows = np.concatenate((windows,5*255+roll.rolling_window(data[i:win_y+i,:,5],(win_y,win_x))),axis=3)
        windows = np.concatenate((windows,6*255+roll.rolling_window(data[i:win_y+i,:,6],(win_y,win_x))),axis=3)
        windows = np.concatenate((windows,7*255+roll.rolling_window(data[i:win_y+i,:,7],(win_y,win_x))),axis=3)
        windows = np.concatenate((windows,8*255+roll.rolling_window(data[i:win_y+i,:,8],(win_y,win_x))),axis=3)
        windows = np.concatenate((windows,9*255+roll.rolling_window(data[i:win_y+i,:,9],(win_y,win_x))),axis=3)
        windows = np.squeeze(windows)
        for j in range(windows.shape[0]):
            features[i+int(win_y/2)+1,j+int(win_x/2)+2,0:8*D] = np.squeeze(cv2.calcHist([windows[i,:,:]],[0],None,[8*D],[0,256*D]))
            features[i+int(win_y/2)+1,j+int(win_x/2)+2,8*D:(8+1)*D] = data[i+int(win_y/2)+1,j+int(win_x/2)+2,:]
    return features

#%% Classification

def validation_split(features,labels,valPercent):
    ''' Split into validation and training sets
        splits features and labels passed in as NxN matricies
        returns validation and training matricies as N*N x 1 array with random order
    '''
    fN1,fN2,fC = np.shape(features)
    lN1,lN2 = np.shape(labels)
    #Data and labels must have same shape
    assert fN1 == lN1
    assert fN2 == lN2    
    #place features and labels into flat arrays
    features = np.reshape(features,(fN1*fN2,fC))
    labels = labels.flatten()
    #shuffle samples preserving labels
    order = list(range(np.shape(features)[0]))
    np.random.shuffle(order)
    features = features[order,:]
    labels = labels[order]
    #split data according to valPercent
    features_train = features[0:int(fN1*fN2*(1-valPercent))]
    features_val = features[int(fN1*fN2*(1-valPercent))+1:(fN1*fN2)]
    labels_train = labels[0:int(fN1*fN2*(1-valPercent))]
    labels_val = labels[int(fN1*fN2*(1-valPercent))+1:(fN1*fN2)]
    return features_val,features_train,labels_val,labels_train

def lda_init(features, labels):
    
    #determine total number of background labels
    bgPix = np.where(labels == -1)
    bgPixCount = np.size(bgPix)/3
    
    labels = np.ravel(labels)
    
    #randomely order label and feature data for downsampling background
    labels,features = unison_shuffled_copies(labels,features)
    
    #Remove 99.3% of background data for training
    count = 0;
    mask = np.ones(len(labels),dtype = bool)
    for i in range(len(labels)):
        if labels[i] == -1 and count < (0.993 * bgPixCount):
            mask[i] = False
            count +=1
    labels = labels[mask,...]
    no_bgdata = features[mask,...]
        
    return no_bgdata, labels

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

#%% Scoring

def get_f1_score(predictLabelPath,actualLabelPath,r):
    predict = np.loadtxt(predictLabelPath).astype(np.int16)
    actual = np.loadtxt(actualLabelPath).astype(np.int16)
    max_u16,r2 = 2**16-1,r**2
    xcord,ycord,label = 0,1,2
    for alarm in range(predict.shape[0]):
        nearest_dist,nearest_ind = max_u16,max_u16
        for truth in range(actual.shape[0]):
            if(np.all(actual[truth,[xcord,ycord]]!=[max_u16,max_u16])):
                if(actual[truth,label]==predict[alarm,label]):
                    d2 = (actual[truth,xcord]-predict[alarm,xcord])**2 + \
                         (actual[truth,ycord]-predict[alarm,ycord])**2
                    if(d2 < nearest_dist):
                        nearest_dist = d2
                        nearest_ind = truth
        if(nearest_dist < r2):
            predict[alarm,[xcord,ycord]] = [max_u16,max_u16]
            actual[nearest_ind,[xcord,ycord]] = [max_u16,max_u16]
    f1 = np.zeros(np.unique([actual]).shape[0]-1)
    for i in range(1,np.unique([actual]).shape[0]-1):
        tot_predict = np.sum(predict[:,label]==i)
        tot_actual = np.sum(actual[:,label]==i)
        true_pos = np.sum(np.logical_and(predict[:,xcord]==-1,predict[:,label]==i))
        false_pos = tot_predict-true_pos
        false_neg = tot_actual-true_pos
        f1[i-1] = 2*true_pos/(2*true_pos+false_pos+false_neg)
    
    predict = np.loadtxt(predictLabelPath).astype(np.int16)
    actual = np.loadtxt(actualLabelPath).astype(np.int16)
    predict = predict[predict[:,label]==4]
    actual = actual[actual[:,label]==4]
    for alarm in range(predict.shape[0]):
        for truth in range(actual.shape[0]):
            if(np.all(actual[truth,:]==predict[alarm,:])):
                predict[alarm,[xcord,ycord]] = [max_u16,max_u16]
                actual[truth,[xcord,ycord]] = [max_u16,max_u16]
    true_pos = np.sum(np.logical_and(predict[:,xcord]==-1,predict[:,label]==4))
    false_pos = predict.shape[0]-true_pos
    false_neg = actual.shape[0]-true_pos
    f1[3] = true_pos/(true_pos+false_pos+false_neg)
    return .3*(f1[0]+f1[1]+f1[2])+.1*f1[3]

#%%  main

def main():
    
    ## preprocessing ##
    '''
    data,locs,labels,pond_masks = load_train_data()
    hsv = cv2.cvtColor(data,cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(data,cv2.COLOR_RGB2GRAY)[:,:,None]
    canny = get_canny(data)[:,:,None]
    fourier = get_fourier(gray)
    data = np.concatenate((data,hsv,gray,canny,fourier),axis=2)
    hsv,gray,canny,fourier = 0,0,0,0
    '''
    
    ## feature extraction ##
    '''
    data = extract_features(data)
    '''
    
    ## classification ##
    '''
    The next few lines will need to be altered if not reading in pre aggregated data
    '''
    data = np.load('../data/features.npy')
    data = data[:,:,0:24]
    data = data.reshape(data.shape[0]*data.shape[1],data.shape[2])
    
    # Read in previously labeled data
    labels = load_custom_labels('../data/custom_labels.txt', (6250,6250,80))
    
    lda = LinearDiscriminantAnalysis() #initialize lda
    no_bgdata,labels = lda_init(data,labels)
    lda.fit(no_bgdata,labels) #fit lda
    
    predicted_labels = lda.predict(data) # predict labels
    predicted_labels = np.reshape(predicted_labels,(int(predicted_labels.size**(1/2)),int(predicted_labels.size**(1/2))))
    
    '''
    ## visualization ##
    data_train,a,b,c = load_train_data()    
    util.write_pred_data(data_train,predicted_labels)
    '''
    
    ## postprocessing ##
    
    img,locs,labels,pond_masks = load_train_data()
    row,col = img.shape[0],img.shape[1]
    classed = predicted_labels
    for i in range(len(row)):
        for j in range(len(col)):
            #RC
            if (classed[i,j] == 2):
                if (img[i,j,0] >= 106 and img[i,j,0] <= 40 and img[i,j,1] >= 69 and img[i,j,1] >= 40 and img[i,j,2] >= 95 and img[i,j,2] <= 59):
                    classed[i,j] = 2;
                else:
                    classed[i,j] = -1;
            #WC
            elif (classed[i,j] == 1):
                if (img[i,j,0] >= 227 and img[i,j,0] <= 255 and img[i,j,1] >= 237 and img[i,j,1] >= 255 and img[i,j,2] >= 230 and img[i,j,2] <= 255):
                    classed[i,j] = 1;
                else:
                    classed[i,j] = -1;
            #POOL
            elif (classed[i,j] == 3):
                if (img[i,j,0] >= 32 and img[i,j,0] <= 101 and img[i,j,1] >= 135 and img[i,j,1] >= 197 and img[i,j,2] >= 151 and img[i,j,2] <= 204):
                    classed[i,j] = 3;
                else:
                    classed[i,j] = -1;
            #POND
            elif (classed[i,j] == 4):
                if (img[i,j,0] >= 36 and img[i,j,0] <= 90 and img[i,j,1] >= 40 and img[i,j,1] >= 116 and img[i,j,2] >= 43 and img[i,j,2] <= 84):
                    classed[i,j] = 4;
                else:
                    classed[i,j] = -1;
    
    ## visualization ##
    data_train,a,b,c = load_train_data()
    util.write_pred_data(data_train,classed)
    
    ## scoring ##
    '''
    path='../data/labels.txt'
    score = get_f1_score(path,path,11)
    print(score)
    '''
    
if  __name__ == '__main__':
    main()
