import glob
import os
import cv2
import numpy as np
from math import *
import scipy.io
from skimage import color
from skimage.feature import hog
'''
np.random.seed(7)
'''
seed = 42
np.random.seed(seed)

'''
def getDataPath(mainFolder):
    mainDir = os.listdir(mainFolder)
    ID = []
    for subMain in mainDir:
        subFolder = mainFolder + "\\" + subMain
        subDir = glob.glob(subFolder + "\\*IM.mat")
        for matFiles in subDir:
            ID.append(matFiles)
    return np.asarray(ID)

def getClasses(DataPath):
    Y = np.zeros(np.shape(DataPath)[0])
    for i, path in enumerate(DataPath):
        Y[i] = transformLabel(DataPath[i].split('\\')[-2])
    return Y
'''

def myLoadMat(filename):
    X = scipy.io.loadmat(filename)
    ImageRgb = X['ImageRgb'].astype('float32')
    CurveMatrix = X['CurveMatrix'].astype('float32')
#     ImageRgb = ImageRgb - np.min(ImageRgb)
#     ImageRgb = ImageRgb / np.max(ImageRgb)
    ImageRgb = (ImageRgb - np.mean(ImageRgb))/np.std(ImageRgb)
#     CurveMatrix = CurveMatrix - np.min(CurveMatrix)
#     CurveMatrix = CurveMatrix / np.max(CurveMatrix)
    CurveMatrix = (CurveMatrix - np.mean(CurveMatrix))/np.std(CurveMatrix)
#     CurveMatrix = CurveMatrix.transpose()
    return ImageRgb, CurveMatrix, transformLabel(X['Especie'])

def transformLabel(name):
    Labels = ['Anchoveta', 'Merluza', 'Mote', 'Pampanito', 'Sardina Comun']
    return Labels.index(name)

def LoadDataFish(DataPath, n_classes = 5, dim = [(256,256,3),(100,235)]):
    total = len(DataPath)
    Images = np.zeros((total,*dim[0]), dtype=np.float32)
    Curves = np.zeros((total,*dim[1]), dtype=np.float32)
    y = np.zeros((total))
    for i, path in enumerate(DataPath):
        Images[i], Curves[i], y[i] = myLoadMat(path)
    idx = np.arange(total)
    '''
    np.random.shuffle(idx)
    '''
    Images = Images[idx,:,:,:].reshape((total, -1)).astype('float32')
    Curves = Curves[idx,:,:].reshape((total, -1)).astype('float32')
    y = y[idx]
    X = {'Images' : Images, 'Spectral' : Curves}
    Y = y
    return X, np.asarray(Y)

def LoadImageHOG(filename):
    X = scipy.io.loadmat(filename)
    #agrega spectral
    CurveMatrix = X['CurveMatrix'].astype('float32')
    CurveMatrix = (CurveMatrix - np.mean(CurveMatrix))/np.std(CurveMatrix)
    #
    ImageRgb = X['ImageRgb'].astype('float32')
    ImageRgb = (ImageRgb - np.mean(ImageRgb))/np.std(ImageRgb)
    ImageGray = color.rgb2gray(ImageRgb)
    ppc = 16
    '''
    fd, hog_image = hog(ImageGray, orientations=8, pixels_per_cell=(ppc,ppc),cells_per_block=(4, 4),block_norm= 'L2',visualise=True)
    '''
    fd, hog_image = hog(ImageGray, orientations=8, pixels_per_cell=(ppc,ppc),cells_per_block=(4, 4),block_norm= 'L2',visualize=True)
    '''
    return fd, hog_image, transformLabel(X['Especie'])
    '''
    return fd, hog_image, CurveMatrix, transformLabel(X['Especie'])

'''
def LoadHOGFeatures(DataPath):
'''
def LoadHOGFeatures(DataPath, dim = [(256,256,3),(100,235)]):

    total = len(DataPath)

    #nuevo
    Curves = np.zeros((total,*dim[1]))

#     hog_images = []
    hog_features = []
    y = np.zeros((total))
    for i, path in enumerate(DataPath):
        '''
        fd, hog_image, y[i] = LoadImageHOG(path)
        '''
        fd, hog_image, Curves[i], y[i] = LoadImageHOG(path)
#         hog_images.append(hog_image)
        hog_features.append(fd)
    
    idx = np.arange(total)
    '''
    np.random.shuffle(idx)
    '''
    
#     hog_images = np.asarray(hog_images).reshape((total, -1))
    hog_features = np.asarray(hog_features).reshape((total, -1))
    
#     hog_images = hog_images[idx,:]
    hog_features = hog_features[idx,:]   

#     print(hog_features.shape)
#     print(hog_images.shape)
    Y = y[idx]
    
#     return hog_images, hog_features, np.asarray(Y)

    #agrega curves
    Curves = Curves[idx,:,:].reshape((total, -1))
    '''
    return hog_features, np.asarray(Y)
    '''
    return {'Images': hog_features, 'Spectral': Curves }, np.asarray(Y)
    