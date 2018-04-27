#%% EEL 4930 Final Project
# Christian Marin
# Troy Tharpe
# Dean Fortier

import numpy as np
import core, util

from sklearn.discriminant_analysis \
    import LinearDiscriminantAnalysis

#%% Main

def main():
    
    SIMULATION_DOWNSAMPLE = True
    
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
    
    if SIMULATION_DOWNSAMPLE:
        N1, N2 = 1000, 1000
        data = data[:N1,:N2,:]
    
    print('Extracting features...')
    data = core.extractFeatures(data)
    N1, N2, D = np.shape(data)
    
    if SIMULATION_DOWNSAMPLE:
        D, dsD = 16, 16
        data = data[:,:,:dsD]
        labels = labels[:N1,:N2,:]
    data = data.reshape(N1*N2,-1)
    
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
    
    if SIMULATION_DOWNSAMPLE:
        N1, N2 = 600, 600
        data = data[:N1,:N2,:]    
    
    print('Extracting features...')
    data = core.extractFeatures(data)
    N1, N2, D = np.shape(data)
    
    if SIMULATION_DOWNSAMPLE:
        data = data[:,:,:dsD]
    data = data.reshape(N1*N2,-1)

    print('Testing classifier...')
    labels = core.classify(lda, data,  N1, N2)
    
    ''' visualization
    labels = lda.predict(data).reshape(labels, N1, N2)
    dataTrain, a, b, c = core.loadTrainData()    
    util.writePredData(dataTrain, labels)
    '''
    
    #-------------------------------------------------------------------------#
    
    print('Performing post-processing...')
    
    # centers = blobDetect(data) # tested, proven computationally intensive
    
    img = core.quickReadMat()
    for r in range(N1):
        for c in range(N2):
            if (labels[r, c] != -1):
                labels[r, c] = core.valRGB(labels[r, c], img[r,c,:])
    
    #-------------------------------------------------------------------------#
    
    resultsPath = '../data_test/results.txt'
    truthPath = '../data_test/labels_test.txt'
    pondPaths = []
    for i in range(4):
        pondPaths.append('../data_test/pond'+str(i+1)+'.txt')
    print('Saving alarms...')
    core.saveResults(labels, resultsPath)
    print('Calculating score...')
    score = core.getF1Score(resultsPath, truthPath, pondPaths, radius=15)
    print('Total Score = ' + str(score))
    dataPath = '../data_train/data_train.mat'
    dataObjName = 'data_train'
    data = core.loadTestData(dataPath, dataObjName)
    util.writePredData(data, labels)
    
    #-------------------------------------------------------------------------#

#%% Main

if  __name__ == '__main__':
    main()
