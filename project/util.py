#%% EEL 4930 Final Project
# Christian Marin
# Troy Tharpe
# Dean Fortier

import numpy as np
import matplotlib.pyplot as plt
import cv2, fileinput

#%% utilities

def writeTrainData(dataTrain, locs, labels, pondMasks):
    ''' write original, car/pool labeled, and pond masked images as pngs
            dataTrainOriginal.png: original matrix as png
            dataTrainLabeled.png: pixel-level target label mask (0xFF00FF)
    '''
    dataTrainLabels = dataTrain.copy() # copy training image
    dataTrainMasks = dataTrain.copy() # copy training image
    for i in range(len(np.unique(labels))): # unique pixel labels
        y = np.array(locs[labels==(i+1), 1], dtype=int)
        x = np.array(locs[labels==(i+1), 0], dtype=int)
        for j in range(np.size(locs[labels==(i+1)], axis=0)):
            dataTrainLabels[y[j], x[j], :] = [255, 0, 255]
    for i in range(8): # pond masks
        y = pondMasks[i][:, 1]
        x = pondMasks[i][:, 0]
        for j in range(np.size(pondMasks[i], axis=0)):
            dataTrainMasks[y[j], x[j], :] = [255, 0, 255]
    cv2.imwrite('../data/data_train_original.png', dataTrain[:, :, ::-1])
    cv2.imwrite('../data/data_train_labels.png', dataTrainLabels[:, :, ::-1])
    cv2.imwrite('../data/data_train_masks.png', dataTrainMasks[:, :, ::-1])

def testLabelFlags(path, flags):
    ''' verifies given custom label flags do not appear in an image
            path: filepath to original an image (unlabeled/untagged)
            flags: array of 3-tuples RGB
    '''
    im = cv2.imread(path)[:, :, ::-1]
    N1, N2, C = np.shape(im)
    im = im.reshape((N1*N2, C))
    for f in flags:
        R = np.where(im[:, 0]==f[0])
        G = np.where(im[:, 1]==f[1])
        B = np.where(im[:, 2]==f[2])
        t = np.intersect1d(R, np.intersect1d(G, B))
        if t.size is not 0:
            raise ValueError('Flag ' + f + ' found in image')

def writePredData(dataTrain, labels):
    ''' write labeled image as png
            dataPredLabeled.png: pixel-level target label mask (0xFF00FF)
    '''
    dataPredLabels = dataTrain.copy() # copy training image
    N1, N2 = np.shape(labels)
    for i in range(N1):
        for j in range(N2):
            print(j)
            if labels[i, j] == 1:
               dataPredLabels[i, j, :] = [0, 255, 0]
            elif labels[i, j] == 2:
               dataPredLabels[i, j, :] = [0, 255, 255]
            elif labels[i, j] == 3:
               dataPredLabels[i, j, :] = [255, 255, 0]
            elif labels[i, j] == 4:
               dataPredLabels[i, j, :] = [255, 0, 0]                      
    cv2.imwrite('../data/data_pred_labels.png', dataPredLabels[:, :, ::-1])
    

def writeTrainDataPixels(dataTrain, locs, labels, pondMasks):
    ''' write original and labeled image as png
            dataTrainOriginal.png: original matrix as png
            dataTrainLabeled.png: pixel-level target label mask (0xFF00FF)
    '''
    N1, N2, C = np.shape(dataTrain)
    dataTrainLabels = dataTrain.copy() # copy training image
    # Background class label = -1
    pixelClassLabels = np.array(np.ones((N1, N2)))
    pixelClassLabels *= -1
    for i in range(len(np.unique(labels))): # unique pixel labels
        y = np.array(locs[labels==(i+1), 1], dtype=int)
        x = np.array(locs[labels==(i+1), 0], dtype=int)
        for j in range(np.size(locs[labels==(i+1)], axis=0)):
            for xHalo in range(-4, 4):
                for yHalo in range(-4, 4):
                    dataTrainLabels[y[j] + yHalo, x[j] + xHalo, :] = [255, 0, 255]
                    pixelClassLabels[y[j] + yHalo, x[j] + xHalo] = i +1
                        
    for i in range(8): # pond masks
        y = pondMasks[i][:, 1]
        x = pondMasks[i][:, 0]
        for j in range(np.size(pondMasks[i], axis=0)):
            dataTrainLabels[y[j], x[j], :] = [255, 0, 255]
            pixelClassLabels[y[j], x[j]] = 4
    cv2.imwrite('../data/data_train_original.png', dataTrain[:, :, ::-1])
    cv2.imwrite('../data/data_train_labels.png', dataTrainLabels[:, :, ::-1])
    return pixelClassLabels

def readCustomLabels(inPath, outPath, original=0):
    ''' read in custom data labels from a labeled/tagged image
        outputs text file in format given
            path: path to custom labeled image
            original: path to src image before custom labeling was done
                (defaults to 0)
    '''
    flags = [
        [0, 255, 0], # white car
        [0, 255, 255], # red car
        [255, 255, 0], # pool
        [255, 0, 0], # pond
        # [255, 0, 255], # reserved for existing labels
        # [0, 0, 255], # unused, open to special tag
    ]
    
    if original is not 0:
        print('validating flags...')
        testLabelFlags(original, flags)
        print('flags validated')
    
    im = cv2.imread(inPath)[:, :, ::-1]
    agg = np.asarray(['\n'], dtype='|S11')
    
    print('parsing labels...')
    labelInd = 1
    for f in flags:
        tmp = np.asarray(np.where(im[:, :, 0]==f[0]), dtype='|S4').T
        delim = np.repeat(' ', tmp.shape[0]).astype(dtype='|S1')
        R = np.core.defchararray.add(tmp[:, 1], delim)
        R = np.core.defchararray.add(R, tmp[:, 0])
        
        tmp = np.asarray(np.where(im[:, :, 1]==f[1]), dtype='|S4').T
        delim = np.repeat(' ', tmp.shape[0]).astype(dtype='|S1')
        G = np.core.defchararray.add(tmp[:, 1], delim)
        G = np.core.defchararray.add(G, tmp[:, 0])
        
        tmp = np.asarray(np.where(im[:, :, 2]==f[2]), dtype='|S4').T
        delim = np.repeat(' ', tmp.shape[0]).astype(dtype='|S1')
        B = np.core.defchararray.add(tmp[:, 1], delim)
        B = np.core.defchararray.add(B, tmp[:, 0])
        
        tmp = np.intersect1d(np.intersect1d(R, G), B)
        label = np.repeat(' '+str(labelInd), tmp.shape[0]).astype(dtype='|S2')
        tmp = np.core.defchararray.add(tmp, label)
        if tmp.size is 0:
            tmp = np.asarray(['\n'], dtype='|S11')
        agg = np.concatenate((agg, tmp))
        labelInd += 1
    
    np.savetxt(outPath, agg, fmt='%s', newline='\n')
    with fileinput.FileInput(outPath, inplace=True) as file:
        for line in file:
            if line == 'b\'\\n\'':
                print(line.replace('b\'\\n\'', ''), end='')
            if line == '\\n':
                print(line.replace('\\n', ''), end='')
            print(line.replace('b', '').replace('\'', ''), end='')
    with open(outPath, 'r') as fin:
        data = fin.read().splitlines(True)
    with open(outPath, 'w') as fout:
        fout.writelines(data[1:])
    print('labels written to ' + outPath)
    return agg

def plotTrainLabels(dataTrain, labels, locs):
    ''' scatterplot target labels over training image (given, modified)
    '''
    colors, ll = ['w', 'r', 'b', 'g'], [] # label colors
    plt.figure() # create figure object
    plt.imshow(dataTrain) # add training image (background)
    for i in range(len(np.unique(labels))): # add labels (scatter plotted)
        x = locs[labels==(i+1), 0]
        y = locs[labels==(i+1), 1]
        lbl = plt.scatter(x, y, c=colors[i])
        ll = np.append(ll, lbl)
    plt.legend(ll, ['White Car', 'Red Car', 'Pool', 'Pond'])
    plt.title('Training Data')
    plt.show()

def plotTrainMasks(N, pondMasks, verbose=True):
    ''' generate and plot pond masks over empty figures (given, modified)
            N: image dimension (square)
            pondMasks: list of point labels
    '''
    decodedMasks = np.zeros((N, N, 8+1)) # 0=all, 1-8=standard ponds
    for i in range(8): # for every pond
        x = pondMasks[i][:, 0]
        y = pondMasks[i][:, 1]
        for j in range(np.size(pondMasks[i], axis=0)): # for every pixel label
            decodedMasks[y[j], x[j], 0] = 1 # mark aggregate (0)
            decodedMasks[y[j], x[j], i+1] = 1 # mark individual (1-8)
        if verbose:
            plt.title('Pond Mask '+str(i+1))
            plt.imshow(decodedMasks[:, :, i])
            plt.show()
    plt.title('Pond Masks (All)')
    plt.imshow(decodedMasks[:, :, 0])
    plt.show()

def window(array, window=(0,), aStep=None, wStep=None):
    array = np.asarray(array)
    origShape = np.asarray(array.shape)
    window = np.atleast_1d(window).astype(int)
    aStep = np.ones_like(origShape)
    wStep = np.ones_like(window)
    
    if window.ndim > 1 or np.any(window < 0) or len(array.shape) < len(window):
        raise ValueError('data/window dimensions conflict')
    if np.any(origShape[-len(window):] < window * wStep):
        raise ValueError('data/window dimensions conflict')

    newShape = origShape
    win = window.copy()
    win[win==0] = 1
    
    newShape[-len(window):] += wStep - win * wStep
    newShape = (newShape + aStep - 1) // aStep
    newShape[newShape < 1] = 1
    shape = newShape
    
    strides = np.asarray(array.strides) * aStep
    newStrides = array.strides[-len(window):] * wStep
    newShape = np.concatenate((shape, window))
    newStrides = np.concatenate((strides,newStrides))
    newStrides = newStrides[newShape != 0]
    newShape = newShape[newShape != 0]
    
    return np.lib.stride_tricks.as_strided(array,shape=newShape, strides=newStrides)
