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
    # load data_train.mat: format - <y>,<x>,<r,g,b>
    mat = spio.loadmat('../data/data_train.mat') # load mat object
    data_train = mat.get('data_train') # extract image struct from mat object
    # load labels.txt: format - <x> <y> <label>
    pairs = np.loadtxt('../data/custom_labels.txt').astype(np.uint16) # load label matrix
    locs = pairs[:,0:2] # <x> <y> pixel indices
    labels = pairs[:,2].astype(np.uint8) # <label> 1=whitecar,2=redcar,3=pool,4=pond
    # load pond .txt masks: format - <x> <y>
    pond_masks = []
    for i in range(8):
        file = '../data/pond'+str(i+1)+'.txt'
        pond_masks.append(np.loadtxt(file).astype(np.uint16))
    return data_train,locs,labels,pond_masks

@jit
def load_custom_labels(path,dims):
    ''' load custom training labels (formatted txt from given path)
            path - string to txt file
            dims - shape of features
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
    data = cv2.fastNlMeansDenoising(data,None,10,7,21)
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
        windows = roll.window(data[i:win_y+i,:,0],(win_y,win_x)).astype(np.uint16)
        windows = np.concatenate((windows,1*255+roll.window(data[i:win_y+i,:,1],(win_y,win_x))),axis=3)
        windows = np.concatenate((windows,2*255+roll.window(data[i:win_y+i,:,2],(win_y,win_x))),axis=3)
        windows = np.concatenate((windows,3*255+roll.window(data[i:win_y+i,:,3],(win_y,win_x))),axis=3)
        windows = np.concatenate((windows,4*255+roll.window(data[i:win_y+i,:,4],(win_y,win_x))),axis=3)
        windows = np.concatenate((windows,5*255+roll.window(data[i:win_y+i,:,5],(win_y,win_x))),axis=3)
        windows = np.concatenate((windows,6*255+roll.window(data[i:win_y+i,:,6],(win_y,win_x))),axis=3)
        windows = np.concatenate((windows,7*255+roll.window(data[i:win_y+i,:,7],(win_y,win_x))),axis=3)
        windows = np.concatenate((windows,8*255+roll.window(data[i:win_y+i,:,8],(win_y,win_x))),axis=3)
        windows = np.concatenate((windows,9*255+roll.window(data[i:win_y+i,:,9],(win_y,win_x))),axis=3)
        windows = np.squeeze(windows)
        for j in range(windows.shape[0]):
            features[i+int(win_y/2)+1,j+int(win_x/2)+2,0:8*D] = np.squeeze(cv2.calcHist([windows[i,:,:]],[0],None,[8*D],[0,256*D]))
            features[i+int(win_y/2)+1,j+int(win_x/2)+2,8*D:(8+1)*D] = data[i+int(win_y/2)+1,j+int(win_x/2)+2,:]
    return features

#%% Classification

def validation_split(features,lables,valPercent):
    ''' Split into validation and training sets
        splits features and lables passed in as NxN matricies
        returns validation and training matricies as N*N x 1 array with random order
    '''
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
    return features_val,features_train,lables_val,lables_train

#%% Scoring
    
def score(predictedLabelPath, trueLabelPath, haloRadius):
    ''' Reads in two text files containing labels and halo radius
        Returns aggregated score percentage A
        Where A = 0.3*f1white + 0.3*f1red + 0.3*f1pool + 0.1*f1pond
        where f1 = 2*tp/(2tp + fp + fn)
    '''
    predictedPairs = np.loadtxt(predictedLabelPath).astype(np.uint16) # load label matrix
    predictedLocs = pairs[:,0:2] # <x> <y> pixel indices
    predictedLabels = pairs[:,2].astype(np.uint8) # <label> 1=whitecar,2=redcar,3=pool,4=pond
    
    truePairs = np.loadtxt(trueLabelPath).astype(np.uint16) # load label matrix
    trueLocs = pairs[:,0:2] # <x> <y> pixel indices
    trueLabels = pairs[:,2].astype(np.uint8) # <label> 1=whitecar,2=redcar,3=pool,4=pond
    
def circle(self, x0, y0, radius, colour=1):
    f = 1 - radius
    ddf_x = 1
    ddf_y = -2 * radius
    x = 0
    y = radius
    self.set(x0, y0 + radius, colour)
    self.set(x0, y0 - radius, colour)
    self.set(x0 + radius, y0, colour)
    self.set(x0 - radius, y0, colour)

    while x < y:
      if f >= 0: 
        y -= 1
        ddf_y += 2
        f += ddf_y
        x += 1
        ddf_x += 2
        f += ddf_x    
        self.set(x0 + x, y0 + y, colour)
        self.set(x0 - x, y0 + y, colour)
        self.set(x0 + x, y0 - y, colour)
        self.set(x0 - x, y0 - y, colour)
        self.set(x0 + y, y0 + x, colour)
        self.set(x0 - y, y0 + x, colour)
        self.set(x0 + y, y0 - x, colour)
        self.set(x0 - y, y0 - x, colour)
        Bitmap.circle = circle

        bitmap = Bitmap(25,25)
        bitmap.circle(x0=12, y0=12, radius=12)
        bitmap.chardisplay()
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
    
    # classification
    data = np.load('../data/data_temp_features.npy')
    load_custom_labels(path,dims):
    data = data.reshape(data.shape[0]*data.shape[1],data.shape[2])
    lda = LinearDiscriminantAnalysis()
    lda.fit(data,np.ravel(labels))
    X = lda.transform(data)
    
    
    '''
    whitePix = np.where(pixel_class_labels == 1)
    redPix = np.where(pixel_class_labels == 2)
    poolPix = np.where(pixel_class_labels == 3)
    pondPix = np.where(pixel_class_labels == 4)
    bgPix = np.where(pixel_class_labels == -1)
    bgPixCount = np.size(bgPix)/2
    
    #Split into training and validation
    features_val,features_train,lables_val,lables_train = validation_split(features,pixel_class_labels,0)
    
    #Remove 99.9% of background data for training
    count = 0;
    mask = np.ones(len(lables_train),dtype = bool)
    for i in range(len(lables_train)):
        if lables_train[i] == -1 and count < (0.995 * bgPixCount):
            mask[i] = False
            count +=1
    nobg_lables_train = lables_train[mask,...]
    nobg_features_train = features_train[mask,...]
    
    #Remove all of background data for testing
    count = 0;
    mask = np.ones(len(lables_train),dtype = bool)
    for i in range(len(lables_train)):
        if lables_train[i] == -1:
            mask[i] = False
            count +=1
    nobg_lables_test = lables_train[mask,...]
    nobg_features_test = features_train[mask,...]
    
    #Predict lables or probability
    predicted_lables = lda.predict(features_train)
    
    lda.fit(nobg_features_train,nobg_lables_train)
    lda.fit(features_train,lables_train)
    lda.predict([[-0.8,-1]])
    
    predicted_lables = np.reshape(predicted_lables,(int(predicted_lables.size**(1/2)),int(predicted_lables.size**(1/2))))
    write_pred_data(data_train,predicted_lables)
    '''

    #Score
    '''
    f1score = f1_score(nobg_lables_train,predicted_lables,average = 'micro')
    #f1score = f1_score(nobg_lables_test,predicted_lables,average = 'micro')
    print('f1 score = ' + f1score)
    '''

if  __name__ == '__main__':
    main()
#%%  Setup for Scoring Tests   
    
    data = np.load('../data/data_temp_features.npy')  
    data = data.reshape(data.shape[0]*data.shape[1],data.shape[2])
#%%  Score Testing space
    
    trueLabelPath = '../data/labels.txt'
    #true_labels_mask = load_custom_labels(trueLabelPath,(6250,6250,1))
    pairs = np.loadtxt(trueLabelPath).astype(np.uint16) # load label matrix
    locs = pairs[:,0:2] # <x> <y> pixel indices
    labels = pairs[:,2].astype(np.uint8) # <label> 1=whitecar,2=redcar,3=pool,4=pond
    numPoints, dim = locs.shape
    for(n in range numPoints):
        for(i in range)
    
    #lda = LinearDiscriminantAnalysis()
    #lda.fit(data,np.ravel(labels))
    
    
    
    