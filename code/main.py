#%% EEL 4930 Final Project
# Christian Marin
# Troy Tharpe
# Dean Fortier

import numpy as np # (given)
import matplotlib.pyplot as plt # (given)
import scipy.io as spio # (given)

import cv2

from skimage.feature import greycomatrix, greycoprops
from skimage import data
from numba import jit
from roll import rolling_window

#%% data i/o

def read_train_data():
    ''' loads provided training data from 'data' folder in root directory
            data_train: (6250,6250,3) (ypixel,xpixel,r/g/b)
            locs: (112,2) (index,xpixel/ypixel)
            labels: (112,1) (index,label)
            pond_masks: (?,2) (index,xpixel/ypixel)
    '''
    # load data_train.mat: format - <y>,<x>,<r,g,b>
    mat = spio.loadmat('data_train') # load mat object
    data_train = mat.get('data_train') # extract image struct from mat object
    # load labels.txt: format - <x> <y> <label>
    pairs = np.loadtxt('labels.txt').astype(int) # load label matrix
    locs = pairs[:,0:2] # <x> <y> pixel indices
    labels = pairs[:,2] # <label> 1=whitecar,2=redcar,3=pool,4=pond
    # load pond#.txt masks: format - <x> <y>
    pond_masks = []
    for i in range(8):
        file = 'pond'+str(i+1)+'.txt'
        pond_masks.append(np.loadtxt(file).astype(int))
    return data_train,locs,labels,pond_masks


def write_train_data(data_train,locs,labels,pond_masks):
    ''' write original and labeled image as png
            data_train_original.png: original matrix as png
            data_train_labeled.png: pixel-level target label mask (0xFF00FF)
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
    cv2.imwrite('data_train_original.png',data_train[:,:,::-1])
    cv2.imwrite('data_train_labels.png',data_train_labels[:,:,::-1])
    cv2.imwrite('data_train_masks.png',data_train_masks[:,:,::-1])


#%% viualization

def plot_train_labels(data_train,labels,locs):
    ''' scatterplot target labels over training image (given, modified)
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
    ''' generate and plot pond masks over empty figures (given, modified)
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

#%% GLCM Texture Features
# GLCM (Grey Level Co-occurrence Matrices)

def glcm_texture():
    a = 1;
    return a

#%% FFT (HighPass Filter)

def cv_dft(image_name, HPF_size=60, sel_color=0, a=1, b=2):
    if sel_color == 0:
        img = cv2.imread(image_name,0) #convert to gray
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

#%% Image Denoising ("low pass filter")

def denoise_colored_image(image_name):
    img = cv2.imread(image_name)

    dst = cv2.fastNlMeansDenoisingColored(img,None,4,4,3,17)
    
    plt.figure(3)
    plt.subplot(121),plt.imshow(img)
    plt.subplot(122),plt.imshow(dst)
    plt.show()
    cv2.imwrite('data_train_denoised.png', dst[:,:,::-1])
    
def bilateral_filter(image_name):
    img = cv2.imread(image_name)
    cv2.bilateralFilter(img,9,75,75)

#%% feature extraction
@jit
def extract_features(data,win_y,win_x,nfeatures):
    ''' extract features from given image over sliding window
            data - 3D image (2D image x color dimensions)
            win_y - window height
            win_x - window width
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
        features[:,:,feature_iter+0] = np.mean(windows,axis=(2,3))
        features[:,:,feature_iter+1] = np.median(windows,axis=(2,3))
        feature_iter += nfeatures
    return features



#%%  main

def main():
    
    data_train,locs,labels,pond_masks = read_train_data()
    # write_train_data(data_train,locs,labels,pond_masks)
    # plot_train_labels(data_train,labels,locs)
    # plot_train_masks(np.size(data_train,axis=0),pond_masks)
    features = extract_features(data_train,6,6,2)    
    # dft w/o denoising
	dft = cv_dft('data_train_original.png',a=1,b=2)
	# =============================================================================
	# #dft w denoising
	# denoise_colored_image('data_train_denoised.png')
	# denoised_dft = cv_dft('data_train_denoised.png',a=3,b=4)
	# =============================================================================
	#dft w bilateral
	bilateral_filter('data_train_denoised.png')
	denoised_dft = cv_dft('data_train_denoised.png',a=3,b=4)



if  __name__ == '__main__':
	main()

