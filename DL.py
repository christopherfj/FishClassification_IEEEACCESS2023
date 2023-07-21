import glob
import os
import cv2
import matplotlib.pylab as plt
import numpy as np
'''
from ConfusionMatrix import plot_confusion_matrix
from keras.models import Sequential
from keras.models import Model
'''
from tensorflow.keras.models import Sequential, Model
'''
from keras.layers import Input, Conv1D, Dense, MaxPool1D, Dropout, Activation, Flatten, LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
'''
from tensorflow.keras.layers import Input, Conv1D, Dense, MaxPool1D, Dropout, Activation, Flatten, LeakyReLU
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import BatchNormalization
'''
from keras.callbacks import EarlyStopping, ModelCheckpoint
'''
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
'''
from keras.optimizers import SGD, Adam
'''
from tensorflow.keras.optimizers import SGD, Adam
from keras.utils.np_utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras import backend as K
from sklearn.preprocessing import StandardScaler
from math import *
import scipy.io
#from ConfusionMatrix import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

seed = 42
np.random.seed(seed)
import random
import tensorflow as tf

#reproducible results
def reproducible_results(seed):  
  np.random.seed(seed)
  random.seed(seed)
  os.environ['PYTHONHASHSEED']=str(0)
  tf.compat.v1.set_random_seed(seed)


def getDataPath(mainFolder):
    mainDir = os.listdir(mainFolder)
    ID = []
    for subMain in mainDir:
        #subFolder = mainFolder + "\\" + subMain
        subFolder = mainFolder + "//" + subMain
        #subDir = glob.glob(subFolder + "\\*IM.mat")
        subDir = glob.glob(subFolder + "//*IM.mat")
        for matFiles in subDir:
            ID.append(matFiles)
    #return np.asarray(ID)
    return np.sort(np.asarray(ID))

def getClasses(DataPath):
    Y = np.zeros(np.shape(DataPath)[0])
    for i, path in enumerate(DataPath):
        
        #server
        Y[i] = transformLabel(DataPath[i].split('\\')[-2])
        #Y[i] = transformLabel(DataPath[i].split('//')[-2])
    return Y

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

#My data generator to train the CNN
def LoadDataFish(DataPath, n_classes = 5, dim = [(256,256,3),(100,235)]):
    total = len(DataPath)
    #Images = np.zeros((total,*dim[0]))
    #Curves = np.zeros((total,*dim[1]))
    Images = np.zeros((total,*dim[0]), dtype=np.float32)
    Curves = np.zeros((total,*dim[1]), dtype=np.float32)
    
    y = np.zeros((total))
    for i, path in enumerate(DataPath):
        [Images[i], Curves[i], y[i]] = myLoadMat(path)
    idx = np.arange(total)
    '''
    np.random.shuffle(idx)
    '''
    Images = Images[idx,:,:,:]
    Curves = Curves[idx,:,:]
    y = y[idx]
    X = [Images, Curves]
    Y = to_categorical(y,n_classes)
    return X, Y
    
    
'''
def defineModelo(params, dim_data_img=(256,256,3), dim_data_matrix=(100,235), num_classes=5):
'''
def defineModelo(params, tipo, dim_data_img=(256,256,3), dim_data_matrix=(100,235), num_classes=5, seed=seed):
  reproducible_results(seed)
  print(params.keys())
  kernel_size = params['kernel_size']
  kernel_initializer = params['kernel_initializer']
  pool_size = params['pool_size']
  DropProbability = params['dropProbability']
  activationUse = params['activationUse']
  dense = params['dense']
  n_filters = params['n_filters']

  # first input model (rgb)
  visible0 = Input(shape=dim_data_img)
  conv00 = Conv2D(n_filters[0,0], kernel_size=kernel_size[0,0], padding='same', kernel_initializer=kernel_initializer[0])(visible0)
  batchnorm00 = BatchNormalization()(conv00)
  activ00 = Activation(activationUse)(batchnorm00)
  pool00 = MaxPooling2D(pool_size=pool_size[0,0])(activ00)
  '''
  drop00 = Dropout(DropProbability)(pool00)
  '''
  drop00 = Dropout(DropProbability, seed = seed)(pool00)
  conv01 = Conv2D(n_filters[0,1], kernel_size=kernel_size[0,1], padding='same', kernel_initializer=kernel_initializer[0])(drop00)
  batchnorm01 = BatchNormalization()(conv01)
  activ01 = Activation(activationUse)(batchnorm01)
  pool01 = MaxPooling2D(pool_size=pool_size[0,1])(activ01)
  '''
  drop01 = Dropout(DropProbability)(pool01)
  '''
  drop01 = Dropout(DropProbability, seed = seed)(pool01)
  flat0 = Flatten()(drop01)

  # second input model (spec)
  visible1 =  Input(shape=dim_data_matrix)
  conv10 = Conv1D(filters=n_filters[1,0], kernel_size=kernel_size[1,0], padding='same', kernel_initializer=kernel_initializer[1])(visible1)
  batchnorm10 = BatchNormalization()(conv10)
  activ10 = Activation(activationUse)(batchnorm10)
  pool10 = MaxPool1D(pool_size=5)(activ10)
  '''
  drop10 = Dropout(DropProbability)(pool10)
  '''
  drop10 = Dropout(DropProbability, seed = seed)(pool10)
  conv11 = Conv1D(filters=n_filters[1,1], kernel_size=kernel_size[1,1], padding='same', kernel_initializer=kernel_initializer[1])(drop10)
  batchnorm11 = BatchNormalization()(conv11)
  activ11 = Activation(activationUse)(batchnorm11)
  pool11 = MaxPool1D(pool_size=pool_size[1,1])(activ11)
  '''
  drop11 = Dropout(DropProbability)(pool11)
  '''
  drop11 = Dropout(DropProbability, seed = seed)(pool11)
  flat1 = Flatten()(drop11)

  if 'RGB' in tipo:
    # interpretation model
    hidden1 = Dense(dense[0], activation=activationUse)(flat0)
  elif 'SPEC' in tipo:
    # interpretation model
    hidden1 = Dense(dense[0], activation=activationUse)(flat1)
  elif 'BOTH' in tipo:
    # merge input models
    merge = concatenate([flat0, flat1])
    # interpretation model
    hidden1 = Dense(dense[0], activation=activationUse)(merge)

  '''
  drop1 = Dropout(DropProbability)(hidden1)
  '''
  drop1 = Dropout(DropProbability, seed = seed)(hidden1)
  hidden2 = Dense(dense[1], activation=activationUse)(drop1)
  '''
  drop2 = Dropout(DropProbability)(hidden2)
  '''
  drop2 = Dropout(DropProbability, seed = seed)(hidden2)
  hidden3 = Dense(dense[2], activation=activationUse)(drop2)
  '''
  drop3 = Dropout(DropProbability)(hidden3) 
  '''
  drop3 = Dropout(DropProbability, seed = seed)(hidden3)    
  output = Dense(num_classes, activation='softmax')(drop3)
  
  if 'RGB' in tipo:
    model = Model(inputs=[visible0], outputs=output)
  elif 'SPEC' in tipo:
    model = Model(inputs=[visible1], outputs=output)
  elif 'BOTH' in tipo:
    model = Model(inputs=[visible0, visible1], outputs=output)

  # summarize layers
  print(model.summary())
  return model
