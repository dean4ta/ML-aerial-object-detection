#%% EEL 4930 Final Project
# Christian Marin
# Troy Tharpe
# Dean Fortier

import numpy as np # (given)
import matplotlib.pyplot as plt # (given)
import scipy.io as spio # (given)

import cv2, fileinput

def write_train_data(data_train,locs,labels,pond_masks):
    ''' write original, car/pool labeled, and pond masked images as pngs
            data_train_original.png: original matrix as png
            data_train_labeled.png: pixel-level target label mask (0xFF00FF)
        ex) write_train_data(data_train,locs,labels,pond_masks)
    '''
    data_train_labels = data_train.copy() # copy training image
    data_train_masks = data_train.copy() # copy training image
    for i in range(len(np.unique(labels))): # unique pixel labels
        y = np.array(locs[labels==(i+1),1],dtype=int)
        x = np.array(locs[labels==(i+1),0],dtype=int)
        for j in range(np.size(locs[labels==(i+1)],axis=0)):
            data_train_labels[y[j],x[j],:] = [255,0,255]
    for i in range(8): # pond masks
        y = pond_masks[i][:,1]
        x = pond_masks[i][:,0]
        for j in range(np.size(pond_masks[i],axis=0)):
            data_train_masks[y[j],x[j],:] = [255,0,255]
    cv2.imwrite('../data/data_train_original.png',data_train[:,:,::-1])
    cv2.imwrite('../data/data_train_labels.png',data_train_labels[:,:,::-1])
    cv2.imwrite('../data/data_train_masks.png',data_train_masks[:,:,::-1])

def test_label_flags(path, flags):
    ''' verifies given custom label flags do not appear in an image
            path: filepath to original an image (unlabeled/untagged)
            flags: array of 3-tuples RGB
        ex) test_label_flags(original,flags) - expect error or all is good
    '''
    im = cv2.imread(path)[:,:,::-1]
    N1,N2,C = np.shape(im)
    im = im.reshape((N1*N2,C))
    for f in flags:
        R = np.where(im[:,0]==f[0])
        G = np.where(im[:,1]==f[1])
        B = np.where(im[:,2]==f[2])
        t = np.intersect1d(R,np.intersect1d(G,B))
        if t.size is not 0:
            raise ValueError('Flag ' + f + ' found in image')

def write_pred_data(data_train,labels):
    ''' write labeled image as png
            data_pred_labeled.png: pixel-level target label mask (0xFF00FF)
    '''
    data_pred_labels = data_train.copy() # copy training image
    N1, N2 = np.shape(labels)
    for i in range(N1):
        for j in range(N2):
            print(j)
            if labels[i,j] == 1:
               data_pred_labels[i,j,:] = [0,255,0]
            elif labels[i,j] == 2:
               data_pred_labels[i,j,:] = [0,255,255]
            elif labels[i,j] == 3:
               data_pred_labels[i,j,:] = [255,255,0]
            elif labels[i,j] == 4:
               data_pred_labels[i,j,:] = [255,0,0]
                           
    cv2.imwrite('../data/data_pred_labels.png',data_pred_labels[:,:,::-1])
    

def write_train_data_pixels(data_train,locs,labels,pond_masks):
    ''' write original and labeled image as png
            data_train_original.png: original matrix as png
            data_train_labeled.png: pixel-level target label mask (0xFF00FF)
    '''
    N1, N2, C = np.shape(data_train)
    data_train_labels = data_train.copy() # copy training image
    '''
        Background class label = -1
    '''
    pixel_class_labels = np.array(np.ones((N1, N2)))
    pixel_class_labels *= -1

    
    for i in range(len(np.unique(labels))): # unique pixel labels
        y = np.array(locs[labels==(i+1),1],dtype=int)
        x = np.array(locs[labels==(i+1),0],dtype=int)
        for j in range(np.size(locs[labels==(i+1)],axis=0)):
            for x_halo in range(-4,4):
                for y_halo in range(-4,4):
                    data_train_labels[y[j] + y_halo, x[j] + x_halo,:] = [255,0,255]
                    pixel_class_labels[y[j] + y_halo, x[j] + x_halo] = i +1
                        
    for i in range(8): # pond masks
        y = pond_masks[i][:,1]
        x = pond_masks[i][:,0]
        for j in range(np.size(pond_masks[i],axis=0)):
            data_train_labels[y[j],x[j],:] = [255,0,255]
            pixel_class_labels[y[j],x[j]] = 4
    cv2.imwrite('../data/data_train_original.png',data_train[:,:,::-1])
    cv2.imwrite('../data/data_train_labels.png',data_train_labels[:,:,::-1])
    
    return pixel_class_labels

def read_custom_labels(in_path,out_path,original=0):
    ''' read in custom data labels from a labeled/tagged image
        outputs text file in format given
            path: path to custom labeled image
            original: path to src image before custom labeling was done
                (defaults to 0)
        ex) custom_labels = read_custom_labels(labels_png,output_txt,original_png)
    '''
    flags = [
        [0,255,0], # white car
        [0,255,255], # red car
        [255,255,0], # pool
        [255,0,0], # pond
        # [255,0,255], # reserved for existing labels
        # [0,0,255], # unused, open to special tag
    ]
    
    if original is not 0:
        print('validating flags...')
        test_label_flags(original,flags)
        print('flags validated')
    
    im = cv2.imread(in_path)[:,:,::-1]
    agg = np.asarray(['\n'], dtype='|S11')
    
    print('parsing labels...')
    label_ind = 1
    for f in flags:
        tmp = np.asarray(np.where(im[:,:,0]==f[0]),dtype='|S4').T
        delim = np.repeat(' ',tmp.shape[0]).astype(dtype='|S1')
        R = np.core.defchararray.add(tmp[:,1], delim)
        R = np.core.defchararray.add(R, tmp[:,0])
        
        tmp = np.asarray(np.where(im[:,:,1]==f[1]),dtype='|S4').T
        delim = np.repeat(' ',tmp.shape[0]).astype(dtype='|S1')
        G = np.core.defchararray.add(tmp[:,1], delim)
        G = np.core.defchararray.add(G, tmp[:,0])
        
        tmp = np.asarray(np.where(im[:,:,2]==f[2]),dtype='|S4').T
        delim = np.repeat(' ',tmp.shape[0]).astype(dtype='|S1')
        B = np.core.defchararray.add(tmp[:,1], delim)
        B = np.core.defchararray.add(B, tmp[:,0])
        
        tmp = np.intersect1d(np.intersect1d(R,G),B)
        label = np.repeat(' '+str(label_ind),tmp.shape[0]).astype(dtype='|S2')
        tmp = np.core.defchararray.add(tmp, label)
        if tmp.size is 0:
            tmp = np.asarray(['\n'],dtype='|S11')
        agg = np.concatenate((agg,tmp))
        label_ind += 1
    
    np.savetxt(out_path,agg,fmt='%s',newline='\n')
    with fileinput.FileInput(out_path,inplace=True) as file:
        for line in file:
            if line == 'b\'\\n\'':
                print(line.replace('b\'\\n\'',''),end='')
            if line == '\\n':
                print(line.replace('\\n',''),end='')
            print(line.replace('b','').replace('\'',''),end='')
    with open(out_path, 'r') as fin:
        data = fin.read().splitlines(True)
    with open(out_path, 'w') as fout:
        fout.writelines(data[1:])
    print('labels written to ' + out_path)
    return agg

def plot_train_labels(data_train,labels,locs):
    ''' scatterplot target labels over training image (given,modified)
        ex) plot_train_labels(data_train,labels,locs)
    '''
    colors,ll = ['w','r','b','g'],[] # label colors
    plt.figure() # create figure object
    plt.imshow(data_train) # add training image (background)
    for i in range(len(np.unique(labels))): # add labels (scatter plotted)
        x = locs[labels==(i+1),0]
        y = locs[labels==(i+1),1]
        lbl = plt.scatter(x,y,c=colors[i])
        ll = np.append(ll,lbl)
    plt.legend(ll,['White Car','Red Car','Pool','Pond'])
    plt.title('Training Data')
    plt.show()

def plot_train_masks(N,pond_masks):
    ''' generate and plot pond masks over empty figures (given,modified)
            N: image dimension (square)
            pond_masks: list of point labels
        ex) plot_train_masks(np.size(data_train,axis=0),pond_masks)
    '''
    decoded_masks = np.zeros((N,N,8+1)) # 0=all,1-8=standard ponds
    for i in range(8): # for every pond
        x = pond_masks[i][:,0]
        y = pond_masks[i][:,1]
        for j in range(np.size(pond_masks[i],axis=0)): # for every pixel label
            decoded_masks[y[j],x[j],0] = 1 # mark aggregate (0)
            decoded_masks[y[j],x[j],i+1] = 1 # mark individual (1-8)
        plt.title('Pond '+str(i+1))
        plt.imshow(decoded_masks[:,:,i])
        plt.show()
    plt.title('Ponds (All)')
    plt.imshow(decoded_masks[:,:,0])
    plt.show()
