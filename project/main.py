#%% EEL 4930 Final Project
# Christian Marin
# Troy Tharpe
# Dean Fortier

import numpy as np
import core
import util

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
    #↓ downsampled for system demonstration ↓#
    D, dsD = 64, 64
    data = data[:,:,:dsD].reshape(N1*N2,dsD)
    labels = labels[:N1,:N2,:]
    #↑ comment for conceptually relevant simulation ↑#
    
    print('Training classifier...')
    data, labels = core.ldaInit(data, labels)
    '''
    For later visualization
    labels = lda.predict(data).reshape(labels, N1, N2)
    dataTrain, a, b, c = core.loadTrainData()    
    util.writePredData(dataTrain, labels)
    '''
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
    #↓ downsampled for system demonstration ↓#
    data = data[:,:,:dsD].reshape(N1*N2,dsD)
    #↑ comment for conceptually relevant simulation ↑#
    
    print('Testing classifier...')
    labels = lda.predict(data).astype(np.int16).reshape(N1,N2,1)
    
    #-------------------------------------------------------------------------#
    
    print('Performing post-processing...')
    
    ''' TODO findContours?
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
    
    ''' TODO simpleDetector?
    data = core.loadCustomLabels('../data/custom_labels.txt', (6250, 6250, 1))
    data = 255*(data-np.min(data))/np.max(data).astype(np.uint8)
    data[data>0] = 255
    cv2.imwrite('../data/test.png', data)
    data = data[:, :, 0]
    im = cv2.imread('../data/test.jpg', cv2.IMREAD_GRAYSCALE)
    # Set up the detector with default parameters.
    detector = cv2.SimpleBlobDetector()
    # Detect blobs.
    keypoints = detector.detect(im)
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    imWithKeypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # Show keypoints
    cv2.imshow('Keypoints', imWithKeypoints)
    cv2.waitKey(0)
    '''
    
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
    
    #-------------------------------------------------------------------------#

#%% Main

if  __name__ == '__main__':
    main()
    