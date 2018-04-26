#%% EEL 4930 Final Project
# Christian Marin
# Troy Tharpe
# Dean Fortier

import numpy as np
import scipy.io as spio
import util, cv2
from numba import jit

#%% load/save data

@jit
def loadTrainData(dataPath, dataObjName, labelPath, pondPaths):
    ''' loads provided training data from 'data' folder in root directory
            dataPath: string path to image
            labelPath: string path to car/pool pixel labels
            pondPaths: list of string paths to pond masks
            dataTrain: (6250, 6250, 3) (ypixel, xpixel, r/g/b)
            locs: (112, 2) (index, xpixel/ypixel)
            labels: (112, 1) (index, label)
            pondMasks: (?, 2) (index, xpixel/ypixel)
    '''
    # load data_train.mat: format-<y>, <x>, <r, g, b>
    mat = spio.loadmat(dataPath) # load mat object
    dataTrain = mat.get(dataObjName) # extract image struct from mat object
    # load labels.txt: format-<x> <y> <label>
    pairs = np.loadtxt(labelPath).astype(np.uint16) # load label matrix
    locs = pairs[:, 0:2] # <x> <y> pixel indices
    labels = pairs[:, 2].astype(np.uint8) # <label> 1=whitecar, 2=redcar, 3=pool, 4=pond
    # load pond .txt masks: format-<x> <y>
    pondMasks = []
    for i in range(len(pondPaths)):
        pondMasks.append(np.loadtxt(pondPaths[i]).astype(np.uint16))
    util.plotTrainLabels(dataTrain, labels, locs)
    util.plotTrainMasks(dataTrain.shape[0], pondMasks, verbose=False)
    return dataTrain, locs, labels, pondMasks

@jit
def quickReadMat():
    dataPath = '../data_train/data_train.mat'
    dataObjName = 'data_train'
    mat = spio.loadmat(dataPath)
    return mat.get(dataObjName)

@jit
def loadTestData(dataPath, dataObjName):
    ''' loads provided testing data from 'data' folder in root directory
            dataPath: string path to image
            dataTest: (6250, 6250, 3) (ypixel, xpixel, r/g/b)
    '''
    # load data_test.mat: format-<y>, <x>, <r, g, b>
    mat = spio.loadmat(dataPath) # load mat object
    dataTest = mat.get(dataObjName) # extract image struct from mat object
    return dataTest

@jit
def loadCustomLabels(path, dims):
    ''' load custom training labels (formatted txt from given path)
            path: string to txt file
            dims: shape of features
    '''
    N1, N2, D = dims
    labelsMask = -1*np.ones((N1, N2, 1)).astype(np.uint8)
    labels = np.loadtxt(path).astype(np.uint16)
    locs, labels = labels[:, 0:2], labels[:, 2].astype(np.uint8)
    for i in range(labels.shape[0]):
        if locs[i, 1] < N1 and locs[i, 0] < N2:
            labelsMask[locs[i, 1], locs[i, 0]] = labels[i]
    return labelsMask

def saveResults(labels, path):
    N1, N2, D = labels.shape
    out = np.array([0,0,0]).astype(np.int16).reshape((1,3))
    for i in range(N1):
        for j in range(N2):
            if (labels[i,j] != -1):
                out = np.vstack([out, \
                    np.array([j, i, labels[i, j]]).astype(np.int16)])
    out = np.delete(out, (0), axis=0)
    np.random.shuffle(out)
    np.savetxt(path, out[np.where(out[:,2]!=-1)].astype(np.int16),fmt='%s',newline='\n')

#%% preprocessing

@jit
def getCanny(data, d=9, σColor=75, σSpace=75, minVal=100, maxVal=200):
    ''' performs edge detection using canny filter
            d: diameter of pixel neighborhoods used during filtering
            σColor: color space filter σ (larger -> more distinct colors will blur)
            σSpace: coord space filter σ (larger -> more distant pixels will blur)
            minVal: low threshold (less significant edges are discarded)
            maxVal: high threshold (more significant edges are preserved)
    '''
    return cv2.Canny(cv2.bilateralFilter(data, d, σColor, σSpace), minVal, maxVal, L2gradient=True)

@jit
def getFourier(data, HPFsize=60):
    ''' performs a DFT and high pass filtering
            data: grayscale 2D image array
            HPFsize: High Pass Filter size of box to filter out
    '''
    r, c = int(data.shape[0]/2), int(data.shape[1]/2)
    # data = cv2.fastNlMeansDenoising(data, None, 10, 7, 21)
    data = np.fft.fftshift(cv2.dft(np.float32(data), flags=cv2.DFT_COMPLEX_OUTPUT))
    data[r-HPFsize:r+HPFsize, c-HPFsize:c+HPFsize] = 0
    data = cv2.idft(np.fft.ifftshift(data))
    data = (data/np.max(data)*255)**2
    data[np.where(data>255)] = 255
    return (data).astype(np.uint8)

#%% feature extraction

@jit
def getSpaces(data):
    hsv = cv2.cvtColor(data, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(data, cv2.COLOR_RGB2GRAY)[:, :, None]
    canny = getCanny(data)[:, :, None]
    fourier = getFourier(gray)
    return np.concatenate((data, hsv, gray, canny, fourier), axis=2)

@jit
def extractFeatures(data, winY=15, winX=15, histBins=8):
    data = getSpaces(data)    
    N1, N2, D = np.shape(data)
    typeMaxVal = 255
    features = np.zeros((N1, N2, (histBins+1)*D)).astype(np.uint8)
    nValidRows = N1-winY+1
    for i in range(nValidRows):
        print('iter '+str(i+1)+'/'+str(nValidRows))
        windows = (util.window(data[i:i+winY, :, 0], (winY, winX)).astype(np.uint16))[0,:,:,:]
        for j in range(1, D):
            windows = np.concatenate((windows, j*typeMaxVal+util.window(data[i:i+winY, :, j], (winY, winX))[0,:,:,:]), axis=2)
        for j in range(windows.shape[0]):
            features[i+int(winY/2), j+int(winX/2), 0:histBins*D] = \
                np.squeeze(cv2.calcHist([windows[i, :, :]], [0], None, [histBins*D], [0, (typeMaxVal+1)*D]))
            features[i+int(winY/2), j+int(winX/2), histBins*D:(histBins+1)*D] = \
                data[i+int(winY/2), j+int(winX/2), :]
    return features

#%% classification

@jit
def validationSplit(features, labels, valPercent):
    ''' split into validation and training sets
            splits features and labels passed in as NxN matricies
            returns validation and training matricies as N*N x 1 array with random order
    '''
    fN1, fN2, fC = np.shape(features)
    lN1, lN2 = np.shape(labels)
    # data and labels must have same shape
    assert fN1 == lN1
    assert fN2 == lN2    
    # place features and labels into flat arrays
    features = np.reshape(features, (fN1*fN2, fC))
    labels = labels.flatten()
    # shuffle samples preserving labels
    order = list(range(np.shape(features)[0]))
    np.random.shuffle(order)
    features = features[order, :]
    labels = labels[order]
    # split data according to valPercent
    featuresTrain = features[0:int(fN1*fN2*(1-valPercent))]
    featuresVal = features[int(fN1*fN2*(1-valPercent))+1:(fN1*fN2)]
    labelsTrain = labels[0:int(fN1*fN2*(1-valPercent))]
    labelsVal = labels[int(fN1*fN2*(1-valPercent))+1:(fN1*fN2)]
    return featuresVal, featuresTrain, labelsVal, labelsTrain

@jit
def ldaInit(features, labels):
    # determine total number of background labels
    bgPix = np.where(labels == -1)
    bgPixCount = np.size(bgPix)/3
    labels = np.ravel(labels)
    # randomly order label and feature data for downsampling background
    labels, features = unionShuffledCopies(labels, features)
    # remove 99.3% of background data for training
    count = 0;
    mask = np.ones(len(labels), dtype = bool)
    for i in range(len(labels)):
        if labels[i] == -1 and count < (0.993 * bgPixCount):
            mask[i] = False
            count +=1
    labels = labels[mask, ...]
    noBgData = features[mask, ...]
    return noBgData, labels

@jit
def unionShuffledCopies(a,b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

@jit
def valRGB(labelVal, imgVal):
    if (labelVal == 2):
        if (imgVal[0] >= 106 and imgVal[0] <= 40 and
            imgVal[1] >= 69 and imgVal[1] >= 40 and
            imgVal[2] >= 95 and imgVal[2] <= 59):
            labelVal = 2;
        else:
            labelVal = -1;
    elif (labelVal == 1):
        if (imgVal[0] >= 227 and imgVal[0] <= 255 and
            imgVal[1] >= 237 and imgVal[1] >= 255 and
            imgVal[2] >= 230 and imgVal[2] <= 255):
            labelVal = 1;
        else:
            labelVal = -1;
    elif (labelVal == 3):
        if (imgVal[0] >= 32 and imgVal[0] <= 101 and
            imgVal[1] >= 135 and imgVal[1] >= 197 and
            imgVal[2] >= 151 and imgVal[2] <= 204):
            labelVal = 3;
        else:
            labelVal = -1;
    elif (labelVal == 4):
        if (imgVal[0] >= 36 and imgVal[0] <= 90 and
            imgVal[1] >= 40 and imgVal[1] >= 116 and
            imgVal[2] >= 43 and imgVal[2] <= 84):
            labelVal = 4;
        else:
            labelVal = -1;
    return labelVal

#%% scoring

def getF1Score(predictLabelPath, actualLabelPath, radius=11):
    predict = np.loadtxt(predictLabelPath).astype(np.int16)
    actual = np.loadtxt(actualLabelPath).astype(np.int16)
    maxU16, r2 = 2**16-1, radius**2
    xcord, ycord, label = 0, 1, 2
    for alarm in range(predict.shape[0]):
        nearestDist, nearestInd = maxU16, maxU16
        for truth in range(actual.shape[0]):
            if(np.all(actual[truth, [xcord, ycord]]!=[maxU16, maxU16])):
                if(actual[truth, label]==predict[alarm, label]):
                    d2 = (actual[truth, xcord]-predict[alarm, xcord])**2 + \
                         (actual[truth, ycord]-predict[alarm, ycord])**2
                    if(d2 < nearestDist):
                        nearestDist = d2
                        nearestInd = truth
        if(nearestDist < r2):
            predict[alarm, [xcord, ycord]] = [maxU16, maxU16]
            actual[nearestInd, [xcord, ycord]] = [maxU16, maxU16]
    f1 = np.zeros(np.unique([actual]).shape[0]-1).astype(np.uint64)
    for i in range(1, np.unique([actual]).shape[0]-1):
        nPredict = np.sum(predict[:, label]==i).astype(np.int16)
        nActual = np.sum(actual[:, label]==i).astype(np.int16)
        truePos = np.sum( \
            np.logical_and(predict[:, xcord]==-1, predict[:, label]==i) \
        ).astype(np.int16)
        falsePos = (nPredict-truePos).astype(np.int16)
        falseNeg = (nActual-truePos).astype(np.int16)
        f1[i-1] = 2*truePos/(2*truePos+falsePos+falseNeg+.00000000001)
    predict = np.loadtxt(predictLabelPath).astype(np.int16)
    actual = np.loadtxt(actualLabelPath).astype(np.int16)
    predict = predict[predict[:, label]==4]
    actual = actual[actual[:, label]==4]
    for alarm in range(predict.shape[0]):
        for truth in range(actual.shape[0]):
            if(np.all(actual[truth, :]==predict[alarm, :])):
                predict[alarm, [xcord, ycord]] = [maxU16, maxU16]
                actual[truth, [xcord, ycord]] = [maxU16, maxU16]
    truePos = np.sum(np.logical_and(predict[:, xcord]==-1, predict[:, label]==4))
    falsePos = predict.shape[0]-truePos
    falseNeg = actual.shape[0]-truePos
    f1[3] = truePos/(truePos+falsePos+falseNeg+.00000000001)
    return .3*(f1[0]+f1[1]+f1[2])+.1*f1[3]
