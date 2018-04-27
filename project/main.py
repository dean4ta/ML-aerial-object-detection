#%% EEL 4930 Final Project
# Christian Marin
# Troy Tharpe
# Dean Fortier

import numpy as np
import core

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
    
    ''' findContours
    data = loadCustomLabels('../data/custom_labels.txt', (6250, 6250, 1))
    # data_orig, a, b, c = core.loadTrainData()
    data = 255*(data-np.min(data))/np.max(data).astype(np.uint8)
    data[data>0] = 255
    data = cv2.GaussianBlur(data, (1, 1), 0).astype(np.uint8)
    cv2.imwrite('../data/test.png', data)
    # img = cv2.imread('../data/data_train_original.png')
    cnts = cv2.findContours(data.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts = cnts[1]
    img = data.copy()
    # pairs = np.array(3)
    for c in cnts:
    	# compute the center of the contour
    	M = cv2.moments(c)
    	cX = int(M['m10'] / M['m00'])
    	cY = int(M['m01'] / M['m00'])
     
    	# draw the contour and center of the shape on the image
    	cv2.drawContours(img, [c], -1, (0, 255, 0), 2)
    	cv2.circle(img, (cX, cY), 7, (255, 0, 255), -1)
    	cv2.putText(img, 'center', (cX - 20, cY - 20), 
    	cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        
        # temp = np.array([cX, cY, data_orig[cY, cX, 0]]) 
    	# pairs = np.append(pairs, [temp])
        
    	# show the image
    	cv2.imshow('Image', img)
    	cv2.imwrite('../data/centers.png', img)
    '''
    
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
    
    #-------------------------------------------------------------------------#

#%% Main

if  __name__ == '__main__':
    main()
    