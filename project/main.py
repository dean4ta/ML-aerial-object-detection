#%% EEL 4930 Final Project
# Christian Marin
# Troy Tharpe
# Dean Fortier

import numpy as np
import core
import util
import cv2

from sklearn.discriminant_analysis \
    import LinearDiscriminantAnalysis

#%% Main

def main():
    
    #-------------------------------------------------------------------------#
    
    print('Loading training data...')
    dataPath = '../data_train/data_train.mat'
    dataObjName = 'data_train'
    labelPath = '../data_train/labels.txt'
    customLabelsPath = '../data_train/custom_labels.txt'
    pondPaths = []
    for i in range(8):
        pondPaths.append('../data_train/pond'+str(i+1)+'.txt')
    data = core.loadTrainData(dataPath, dataObjName, labelPath, pondPaths)[0]
    labels = core.loadCustomLabels(customLabelsPath, np.shape(data))
    del dataPath, dataObjName, labelPath, pondPaths, customLabelsPath, i
    
    #↓ downsampled for system demonstration ↓#
    N1, N2 = 1000, 1000
    data = data[:N1,:N2,:]
    #↑ comment for conceptually relevant simulation ↑#
    
    
    print('Extracting features...')
    data = core.extractFeatures(data)
    N1, N2, D = np.shape(data)
    np.save('../data_train/features_train.npy',data)
    np.save('../data_train/labels_train.npy',labels)
    
    #↓ downsampled for system demonstration ↓#
    D, dsD = 64, 64
    data = data[:,:,:dsD]
    labels = labels[:N1,:N2,:]
    #↑ comment for conceptually relevant simulation ↑#
    
    data = data.reshape(N1*N2,-1)
    
    print('Training classifier...')
    data, labels = core.ldaInit(data, labels)

    lda = LinearDiscriminantAnalysis().fit(data, labels)
    del N1, N2, D, labels
    
    #-------------------------------------------------------------------------#
    
    print('Loading testing data...')
    dataPath = '../data_test/data_test.mat'
    dataObjName = 'data_test'
    data = core.loadTestData(dataPath, dataObjName)
    del dataPath, dataObjName
    
    #↓ downsampled for system demonstration ↓#
    N1, N2 = 600, 600
    data = data[:N1,:N2,:]
    #↑ comment for conceptually relevant simulation ↑#
    
    
    print('Extracting features...')
    data = core.extractFeatures(data)
    N1, N2, D = np.shape(data)
    
    np.save('../data_test/features_test.npy',data)
    
    #↓ downsampled for system demonstration ↓#
    data = data[:,:,:dsD].reshape(N1*N2,dsD)
    #↑ comment for conceptually relevant simulation ↑#
    
    data = data.reshape(N1*N2,-1)
    print('Testing classifier...')
    labels = lda.predict(data).astype(np.int16).reshape(N1,N2,1)
    np.save('../data_test/labels_pred.npy',labels)
    
    #-------------------------------------------------------------------------#
    
    print('Performing post-processing...')
    
    #proven to be too computationally intensive to run in a reasonable amount of time
    #centers = blobDetect(data)
    
    img = core.quickReadMat()
    for r in range(N1):
        for c in range(N2):
            if (labels[r, c] != -1):
                labels[r, c] = core.valRGB(labels[r, c], img[r,c,:])
    
    #-------------------------------------------------------------------------#
    
    resultsPath = '../data_test/results.txt'
    truthPath = '../data_test/labels_test.txt'
    print('Saving alarms...')
    core.saveResults(labels, resultsPath)
    print('Calculating score...')
    score = core.getF1Score(resultsPath, truthPath, radius=15)
    print('Total Score = ' + str(score))
    dataPath = '../data_train/data_train.mat'
    dataObjName = 'data_train'
    data = core.loadTestData(dataPath, dataObjName)
    util.writePredData(data, labels)
    
    #-------------------------------------------------------------------------#

#%% Main

if  __name__ == '__main__':
    main()
                
 


    