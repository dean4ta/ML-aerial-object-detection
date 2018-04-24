#%% EEL 4930 Final Project
# Christian Marin
# Troy Tharpe
# Dean Fortier

import numpy as np # (given)
import matplotlib.pyplot as plt # (given)
import scipy.io as spio # (given)

import cv2, fileinput

from matplotlib import colors as mcolors
from numba import jit
from roll import rolling_window
from skimage import data
from skimage.feature import greycomatrix, greycoprops
from sklearn.svm import SVC

RGB_FLAG = [255,0,255]


#%% general data i/o

@jit
def read_train_data():
    ''' loads provided training data from 'data' folder in root directory
            data_train: (6250,6250,3) (ypixel,xpixel,r/g/b)
            locs: (112,2) (index,xpixel/ypixel)
            labels: (112,1) (index,label)
            pond_masks: (?,2) (index,xpixel/ypixel)
    '''
    # load data_train.mat: format - <y>,<x>,<r,g,b>
    mat = spio.loadmat('../data/data_train.mat') # load mat object
    data_train = mat.get('data_train') # extract image struct from mat object
    # load labels.txt: format - <x> <y> <label>
    pairs = np.loadtxt('../data/labels.txt').astype(int) # load label matrix
    locs = pairs[:,0:2] # <x> <y> pixel indices
    labels = pairs[:,2] # <label> 1=whitecar,2=redcar,3=pool,4=pond
    # load pond#.txt masks: format - <x> <y>
    pond_masks = []
    for i in range(8):
        file = '../data/pond'+str(i+1)+'.txt'
        pond_masks.append(np.loadtxt(file).astype(int))
    return data_train,locs,labels,pond_masks

@jit
def read_custom_train_data():
    ''' loads provided training data from 'data' folder in root directory
            data_train: (6250,6250,3) (ypixel,xpixel,r/g/b)
            locs: (112,2) (index,xpixel/ypixel)
            labels: (112,1) (index,label)
            pond_masks: (?,2) (index,xpixel/ypixel)
    '''
    # load data_train.mat: format - <y>,<x>,<r,g,b>
    mat = spio.loadmat('../data/data_train.mat') # load mat object
    data_train = mat.get('data_train') # extract image struct from mat object
    # load labels.txt: format - <x> <y> <label>
    pairs = np.loadtxt('../data/custom_labels.txt').astype(int) # load label matrix
    locs = pairs[:,0:2] # <x> <y> pixel indices
    labels = pairs[:,2] # <label> 1=whitecar,2=redcar,3=pool,4=pond
    return data_train,locs,labels

@jit
def write_train_data(data_train,locs,labels,pond_masks):
    ''' write original, car/pool labeled, and pond masked images as pngs
            data_train_original.png: original matrix as png
            data_train_labeled.png: pixel-level target label mask (0xFF00FF)
    '''
    data_train_labels = data_train.copy() # copy training image
    data_train_masks = data_train.copy() # copy training image
    for i in range(len(np.unique(labels))): # unique pixel labels
        y = np.array(locs[labels==(i+1),1],dtype=int)
        x = np.array(locs[labels==(i+1),0],dtype=int)
        for j in range(np.size(locs[labels==(i+1)],axis=0)):
            data_train_labels[y[j],x[j],:] = RGB_FLAG
    for i in range(8): # pond masks
        y = pond_masks[i][:,1]
        x = pond_masks[i][:,0]
        for j in range(np.size(pond_masks[i],axis=0)):
            data_train_masks[y[j],x[j],:] = RGB_FLAG
    cv2.imwrite('../data/data_train_original.png',data_train[:,:,::-1])
    cv2.imwrite('../data/data_train_labels.png',data_train_labels[:,:,::-1])
    cv2.imwrite('../data/data_train_masks.png',data_train_masks[:,:,::-1])

    
def write_pred_data(data_train,labels):
    ''' write labeled image as png
            data_pred_labeled.png: pixel-level target label mask (0xFF00FF)
    '''
    data_pred_labels = data_train.copy() # copy training image
    N1, N2 = np.shape(labels)
    for i in range(N1):
        for j in range(N2):
            if labels[i,j] == 1:
               data_pred_labels[i,j,:] = [0,255,0]
            elif labels[i,j] == 2:
                data_pred_labels[i,j,:] = [0,255,255]
            elif labels[i,j] == 3:
                data_pred_labels[i,j,:] = [255,255,0]
            elif labels[i,j] == 3:
                data_pred_labels[i,j,:] = [255,0,0]
              
               
    cv2.imwrite('../data/data_pred_labels.png',data_pred_labels[:,:,::-1])
    

def get_pixel_class_lables(data_train,locs,labels):
    ''' Assign a lable to each individual pixel, return flattnened 
        array of every pixel lable in image
        Background class label = -1
    '''
    N1, N2, C = np.shape(data_train)
    pixel_class_labels = np.array(np.ones((N1, N2)))
    pixel_class_labels *= -1

    
    for i in range(len(np.unique(labels))): # unique pixel labels
        y = np.array(locs[labels==(i+1),1],dtype=int)
        x = np.array(locs[labels==(i+1),0],dtype=int)
        for j in range(np.size(locs[labels==(i+1)],axis=0)):
            pixel_class_labels[y[j], x[j]] = i + 1
            
    pixel_class_labels = pixel_class_labels
    
    return pixel_class_labels

@jit
def test_label_flags(path, flags):
    ''' verifies given custom label flags do not appear in an image
            path: filepath to original an image (unlabeled/untagged)
            flags: array of 3-tuples RGB
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


def read_custom_labels(path,original=0):
    ''' read in custom data labels from a labeled/tagged image
		outputs text file in format given
    '''
    flags = [
        [0,255,0], # white car
        [0,255,255], # red car
        [255,255,0], # pool
        [255,0,0], # pond
        # [255,0,255], # reserved for existing labels
        # [0,0,255], # unused, open to special tag
    ]
    '''
    if original is not 0:
        print('validating flags...')
        test_for_labeling(original,flags)
        print('flags validated')
    '''
    im = cv2.imread(path)[:,:,::-1]
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
    
    out = '../data/custom_labels.txt'
    np.savetxt(out,agg,fmt='%s',newline='\n')
    with fileinput.FileInput(out,inplace=True) as file:
        for line in file:
            if line == 'b\'\\n\'':
                print(line.replace('b\'\\n\'',''),end='')
            if line == '\\n':
                print(line.replace('\\n',''),end='')
            print(line.replace('b','').replace('\'',''),end='')
    with open(out, 'r') as fin:
        data = fin.read().splitlines(True)
    with open(out, 'w') as fout:
        fout.writelines(data[1:])
    print('labels written to ' + out)
    
#%% viualization

@jit
def plot_train_labels(data_train,labels,locs):
    ''' scatterplot target labels over training image (given,modified)
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

@jit
def plot_train_masks(N,pond_masks):
    ''' generate and plot pond masks over empty figures (given,modified)
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

#%% preprocessing

@jit
def convert_colorspace(data):
    ''' converts rgb input image into desired color spaces
            data - 3D image (2D image x color dimensions)
    '''
    HSV = np.array(mcolors.rgb_to_hsv(data/255)*255).astype(np.uint8)
    GRY = np.array(0.2989*data[:,:,0]+0.5870*data[:,:,1]+0.1140*data[:,:,2]).astype(np.uint8)[:,:,None]
    data = np.concatenate((data,HSV,GRY),axis=2)
    return data

@jit
def denoise_colored_image(img):
    ''' Denoising with Low Pass
    img:    pass RGB image array
    see https://docs.opencv.org/3.3.1/d5/d69/tutorial_py_non_local_means.html
    for more info
    '''
    #img = cv2.imread(image_name)

    dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
    
	# =============================================================================
	#     plt.figure(3)
	#     plt.subplot(121),plt.imshow(img)
	#     plt.subplot(122),plt.imshow(dst)
	#     plt.show()
	#     cv2.imwrite('../data/data_train_denoised.png', dst[:,:,::-1])
	# =============================================================================
    return dst[:,:,::-1]

@jit 
def bilateral_filter(img):
    #img = cv2.imread(image_name)
    bil = cv2.bilateralFilter(img,9,75,75)
    return bil

@jit
def canny(data,d,sigmaColor,sigmaSpace,minVal,maxVal):
    ''' performs edge detection using canny filter
            d: diameter of pixel neighborhoods used during filtering
            sigmaColor: color space filter sigma
                (larger values, more distinct colors will blur)
            sigmaSpace: coordinate space filter sigma
                (larger value, farther pixels will blur)
            minVal: lower threshold (less significant edges are discarded)
            maxVal: higher threshold (more significant edges are preserved)
    '''
    smooth = cv2.bilateralFilter(data,d,sigmaColor,sigmaSpace)
    edges = cv2.Canny(smooth,minVal,maxVal,L2gradient=True)
    data[edges>0,:] = RGB_FLAG
    cv2.imwrite('../data/canny.png',data[:,:,::-1])

@jit
def cv_dft(img, HPF_size=60, sel_color=0, a=1, b=2):
    ''' FFT with High Pass
		img:        grayscale 2D image array
		HPF_size:   High Pass Filter size of box to filter out (See magnitude spectrum)
		sel_color:  Choose 0 for grayscale. Other reserved for fututre use
		a:          Plot magnitude spectrum on plt.figure(a)
		b:          Plot FT result on plt.figure(b)
    '''
    
    if sel_color == 0:
        #img = cv2.imread(image_name,0) #convert to gray
        dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        
        magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
        
        plt.figure(a)
        plt.subplot(121),plt.imshow(img, cmap = 'gray')
        plt.title('Input Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
        plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
        plt.show()
        
        rows, cols = img.shape
        crow,ccol = int(rows/2) , int(cols/2)
        
        # create a mask first, center square is 1, remaining all zeros
        mask = np.ones((rows,cols,2),np.uint8)
        mask[crow-HPF_size:crow+HPF_size, ccol-HPF_size:ccol+HPF_size] = 0
        
        # apply mask and inverse DFT
        fshift = dft_shift*mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
        
        img_back = img_back/np.max(img_back)*255
        
        img_back = img_back**1.7
        img_back255 = np.where(img_back > 255)
        img_back[img_back255] = 255
        
        plt.figure(b)
        plt.subplot(121),plt.imshow(img, cmap = 'gray')
        plt.title('Input Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(img_back, cmap = 'gray')
        plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
        plt.show()
    
    return img_back

#%% feature extraction

@jit
def extract_features(data,win_y,win_x,nfeatures):
    ''' extract features from given image over sliding window
            data - 3D image (2D image x color dimensions)
            win_y - window height --MUST BE ODD
            win_x - window width  --MUST BE ODD
    '''
    print('Running Feature Extractor...')
    N1,N2,C = np.shape(data)
    features = np.zeros((N1-win_y+1,N2-win_x+1,C*nfeatures))
    feature_iter = 0
    for x in range(C):
        print(' computing sliding windows for',x,'color dimension')
        windows = rolling_window(data[:,:,x],(win_y,win_x))
        # ***** modify below - add features as desired ***** #
        # features[:,:,feature_iter+n] = f(windows,axis=(2,3)) #
        print(' computing features for',x,'color dimension')
        # features[:,:,feature_iter+0] = np.mean(windows,axis=(2,3)) -- update to be self pixel
        features[:,:,feature_iter+0] = np.mean(windows,axis=(2,3))
        features[:,:,feature_iter+1] = np.median(windows,axis=(2,3))
        feature_iter += nfeatures
    return features

#%%  main

def main():
    
	labeled_path='../data/data_train_custom_filled.png'
	read_custom_labels(labeled_path)
	data_train,locs,labels = read_custom_train_data()
	data_train = convert_colorspace(data_train)
    
	pixel_class_lables = get_pixel_class_lables(data_train,locs,labels)
	features = extract_features(data_train,6,6,2)
    
	features = np.pad(features,((3,2),(3 ,2),(int(0), int(0))),mode = 'edge')
	features = np.reshape(features,(6250*6250,14))
	pixel_class_lables = pixel_class_lables.ravel()
  
	clf = SVC()
	clf.fit(features,pixel_class_lables)
    
	predicted_lables = clf.predict(features)
	predicted_lables = predicted_lables.reshape(6250,6250)
	write_pred_data(data_train,predicted_lables)
    

	# canny(data_train.copy(),9,75,75,100,200)
    # write_train_data(data_train,locs,labels,pond_masks)
    # plot_train_labels(data_train,labels,locs)
    # plot_train_masks(np.size(data_train,axis=0),pond_masks)
    # features = extract_features(data_train,6,6,2)

    # dft = cv_dft('../data/data_train_original.png',a=1,b=2) # dft w/o denoising
    # denoise_colored_image('data_train_denoised.png')
    # denoised_dft = cv_dft('data_train_denoised.png',a=3,b=4) # #dft w denoising
	
    # bilateral_filter('../data/data_train_denoised.png')
    # denoised_dft = cv_dft('../data/data_train_denoised.png',a=3,b=4) #dft w bilateral

	# r, g, b = data_train[:,:,0], data_train[:,:,1], data_train[:,:,2]
	# data_gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
	# data_gray = (np.round(data_gray)).astype(np.uint8) #convert to grayscale

	# data_dft = cv_dft(data_gray,a=1,b=2) # Get FT preprocess data

	# denoised_data = denoise_colored_image(data_train)
	# r, g, b = denoised_data[:,:,0], denoised_data[:,:,1], denoised_data[:,:,2]
	# denoised_data = 0.2989 * r + 0.5870 * g + 0.1140 * b
	# denoised_data = (np.round(denoised_data)).astype(np.uint8)

	# denoised_data_dft = cv_dft(denoised_data,a=3,b=4)
	# denoised_data_dft = (np.round(denoised_data_dft)).astype(np.uint8) #cast
	# denoised_data_dft = denoised_data_dft[:,:,None] #make 3D

	# colorData = np.concatenate((data_train, denoised_data_dft), axis=2) #add to color space
	
	# win_y, win_x, nfeatures = 7,7,2
	# features = extract_features(colorData,win_y,win_x,nfeatures) # Extract Features from Color Space

if  __name__ == '__main__':
	main()
