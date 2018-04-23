#%% EEL 4930 Final Project
# Christian Marin
# Troy Tharpe
# Dean Fortier

import numpy as np # (given)
import matplotlib.pyplot as plt # (given)
import scipy.io as spio # (given)
import copy,cv2,util,roll
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
    features = np.zeros((N1,N2,8*D)).astype(np.uint8)
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
            features[i+int(win_y/2)+1,j+int(win_x/2)+2,:] = np.squeeze(cv2.calcHist([windows[i,:,:]],[0],None,[8*D],[0,256*D]))
    return features

#%% Classification

def validation_split(features, lables, valPercent):
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
    return features_val,features_train, lables_val, lables_train


#%%  main

def main():
    
    '''
    data,locs,labels,pond_masks = load_train_data()
    hsv = cv2.cvtColor(data,cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(data,cv2.COLOR_RGB2GRAY)[:,:,None]
    canny = get_canny(data)[:,:,None]
    fourier = get_fourier(gray)
    data = np.concatenate((data,hsv,gray,canny,fourier),axis=2)
    hsv,gray,canny,fourier = 0,0,0,0
    np.save('../data/data_preproc',data)
    '''
    
    '''
    data = np.load('../data/data_preproc.npy')
    extract_features(data)
    np.save('../data/data_features',features)
    '''
    
    features = np.load('../data/data_features.npy')
    
    import numpy as np
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    y = np.array([1, 1, 1, 2, 2, 2])
    clf = LinearDiscriminantAnalysis()
    clf.fit(X, y)
    
    print(clf.predict([[-0.8, -1]]))
    
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
