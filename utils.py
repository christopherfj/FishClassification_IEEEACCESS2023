seed = 42
import os
import numpy as np
from tensorflow.keras.models import load_model
import gc
from collections import defaultdict
from sklearn.svm import SVC

def get_results(train_index, test_index , featureCLF, feature, data, classes, validation_split=None, Epochs=None, batch_size=None, callbacks_list=None, seed = seed):
  if 'DL' in featureCLF:
    ytrain = classes[ train_index ,:]
    ytest = classes[ test_index ,:]
    model = load_model( os.path.join(os.getcwd(), 'out', 'model', 'model_dl.h5') )
    if 'RGB' in featureCLF:
      model.fit( [ data[0][ train_index, :,:,:].astype(np.float32) ], ytrain, validation_split=validation_split, shuffle = True, epochs=Epochs, batch_size=batch_size, callbacks=callbacks_list, verbose=0 )    
      prediction = model.predict( [ data[0][ test_index, :,:,:].astype(np.float32) ] )
    elif 'SPEC' in featureCLF:
      model.fit( [ data[1][ train_index, :,:].astype(np.float32) ], ytrain, validation_split=validation_split, shuffle = True, epochs=Epochs, batch_size=batch_size, callbacks=callbacks_list, verbose=0 )    
      prediction = model.predict( [ data[1][ test_index, :,:].astype(np.float32) ] )
    elif 'BOTH' in featureCLF:
      model.fit( [ data[0][ train_index, :,:,:].astype(np.float32), data[1][ train_index, :,:].astype(np.float32) ], ytrain, validation_split=validation_split, shuffle = True, epochs=Epochs, batch_size=batch_size, callbacks=callbacks_list, verbose=0 )    
      prediction = model.predict( [ data[0][ test_index, :,:,:].astype(np.float32), data[1][ test_index, :,:].astype(np.float32) ] )
    del ytrain
    del model
    gc.collect()
    return np.argmax( ytest, axis = 1 ), np.argmax( prediction, axis = 1 ) 
  else:
    ytrain = classes[ train_index ]
    ytest = classes[ test_index ]
    #SVM
    model = SVC(kernel = 'rbf', random_state = seed)        
    if feature != 'both':
      model.fit( data[feature][ train_index, :].astype(np.float32) , ytrain )
      prediction_svm = model.predict( data[feature][ test_index, :].astype(np.float32)  )  
    else:
      model.fit( np.hstack( ( data['Images'][ train_index, :].astype(np.float32), data['Spectral'][ train_index, :].astype(np.float32) ) )  , ytrain )
      prediction_svm = model.predict( np.hstack( ( data['Images'][ test_index, :].astype(np.float32), data['Spectral'][ test_index, :].astype(np.float32) ) )   ) 
    del model
    gc.collect()
    return ytest, prediction_svm

def get_lc_results( train_index, test_index, featureCLF, feature, training, testing, data, classes, validation_split=None, Epochs=None, batch_size=None, callbacks_list=None, batch=50, seed = seed):
  if 'DL' in featureCLF:
    ytest = classes[ test_index ,:]
    ytrain_aux = np.argmax( classes[ train_index ,:], axis = 1 )
    lc_results = defaultdict(list)
    n_classes = len(list(set(np.argmax( classes, axis = 1 ))))
  else:
    ytest = classes[ test_index ]
    ytrain_aux = classes[ train_index ].copy()
    lc_results_svm = defaultdict(list)
    lc_results_knn = defaultdict(list)
    n_classes = len(list(set(classes)))
  
  indexes = np.arange( len(ytrain_aux) )
  clases_ = ytrain_aux[:batch]        
  while len(set(list(clases_))) != n_classes:
      indexes = shuffle(indexes, random_state = seed)
      clases_ = ytrain_aux[ indexes ][:batch].copy()
  del clases_

  for index in range(0, len(ytrain_aux) , batch):
    print(index)

    if 'DL' in featureCLF:
      ytrain = classes[train_index[indexes][:index+batch], :]
      model = load_model( os.path.join(os.getcwd(), 'out', 'model', 'model_dl.h5') )
      
      if 'RGB' in featureCLF:
        model.fit( [ data[0][train_index[indexes][:index+batch], :,:,:].astype(np.float32) ], ytrain, validation_split=validation_split, shuffle = True, epochs=Epochs, batch_size=batch_size, callbacks=callbacks_list, verbose=0 )
      elif 'SPEC' in featureCLF:
        model.fit( [ data[1][train_index[indexes][:index+batch], :,:].astype(np.float32) ], ytrain, validation_split=validation_split, shuffle = True, epochs=Epochs, batch_size=batch_size, callbacks=callbacks_list, verbose=0 )
      elif 'BOTH' in featureCLF:
        model.fit( [ data[0][train_index[indexes][:index+batch], :,:,:].astype(np.float32), data[1][train_index[indexes][:index+batch], :,:].astype(np.float32) ], ytrain, validation_split=validation_split, shuffle = True, epochs=Epochs, batch_size=batch_size, callbacks=callbacks_list, verbose=0 )

      if training:
        if 'RGB' in featureCLF:
          prediction_tr_dl = model.predict( [ data[0][train_index[indexes][:index+batch], :,:,:].astype(np.float32) ] )
        elif 'SPEC' in featureCLF:
          prediction_tr_dl = model.predict( [ data[1][train_index[indexes][:index+batch], :,:].astype(np.float32) ] )
        elif 'BOTH' in featureCLF:
          prediction_tr_dl = model.predict( [ data[0][train_index[indexes][:index+batch], :,:,:].astype(np.float32), data[1][train_index[indexes][:index+batch], :,:].astype(np.float32) ] ) 
        lc_results['x'].append( ytrain.shape[0] )
        lc_results['training'].append( [np.argmax( ytrain, axis = 1 ), np.argmax( prediction_tr_dl, axis = 1 )] )
      if testing:
        if 'RGB' in featureCLF:
          prediction_test_dl = model.predict( [ data[0][test_index, :,:,:].astype(np.float32) ] )   
        elif 'SPEC' in featureCLF:
          prediction_test_dl = model.predict( [ data[1][test_index, :,:].astype(np.float32) ] )   
        elif 'BOTH' in featureCLF:
          prediction_test_dl = model.predict( [ data[0][test_index, :,:,:].astype(np.float32), data[1][test_index, :,:].astype(np.float32) ] )   
        lc_results['x'].append( ytrain.shape[0] )
        lc_results['test'].append( [np.argmax( ytest, axis = 1 ), np.argmax( prediction_test_dl, axis = 1 )] )
      del model
      del ytrain
      gc.collect()

    else:
      ytrain = classes[train_index[indexes][:index+batch]]
      #svm
      model = SVC(kernel = 'rbf', random_state = seed )
      if feature!= 'both':
        model.fit( data[feature][train_index[indexes][:index+batch], :].astype(np.float32), ytrain )
        if training:
          prediction_tr_svm = model.predict( data[feature][train_index[indexes][:index+batch], :].astype(np.float32) )
          lc_results_svm['x'].append( ytrain.shape[0] )
          lc_results_svm['training'].append( [ytrain, prediction_tr_svm] )
        if testing:
          prediction_test_svm = model.predict( data[feature][test_index, :].astype(np.float32) )
          lc_results_svm['x'].append( ytrain.shape[0] )
          lc_results_svm['test'].append( [ytest, prediction_test_svm] )
      else:
        model.fit( np.hstack( ( data['Images'][train_index[indexes][:index+batch], :].astype(np.float32), data['Spectral'][train_index[indexes][:index+batch], :].astype(np.float32) ) ), ytrain )
        if training:
          prediction_tr_svm = model.predict( np.hstack( ( data['Images'][train_index[indexes][:index+batch], :].astype(np.float32), data['Spectral'][train_index[indexes][:index+batch], :].astype(np.float32) ) ) )
          lc_results_svm['x'].append( ytrain.shape[0] )
          lc_results_svm['training'].append( [ytrain, prediction_tr_svm] )
        if testing:
          prediction_test_svm = model.predict( np.hstack( ( data['Images'][test_index, :].astype(np.float32), data['Spectral'][test_index, :].astype(np.float32) ) ) )
          lc_results_svm['x'].append( ytrain.shape[0] )
          lc_results_svm['test'].append( [ytest, prediction_test_svm] )
      del model
      gc.collect()

  del ytest
  if 'DL' in featureCLF:
    return lc_results
  else:
    return lc_results_svm