from IPython import get_ipython
get_ipython().magic('reset -sf')
import pandas as pd
import numpy as np
from sklearn.utils import resample
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import ConvLSTM2D
from numpy import mean
from numpy import std
from pandas import concat
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
import math 
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA  
from tabulate import tabulate
    
def evaluate_model(y_test,y_pred):
    # analysis of the results by calculating the confusion matrix and defined performance criteria.
    #AUC_test = roc_auc_score(y_test[:,0], y_pred[:,0])    
    tn, fp, fn, tp = confusion_matrix(y_test[:,0], y_pred[:,0]).ravel()
    accuracy_test = 100*(tp + tn)/(tp + tn + fp + fn)
    precision_test = 100*tp/(tp + fp)
    recall_test = 100*tp/(tp + fn)
    f1_test = 100*2*tp/(2*tp + fp + fn)
    MCC_test = ((tp*tn)-(fp*fn))/(math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))    
    
    return accuracy_test, precision_test, recall_test, f1_test, MCC_test
    
        
# fit and evaluate a model
def traintest_convlstm(X, r, delay, n_features, ftr_ly_01, ftr_ly_02, krn_ly_01, krn_ly_02, dropout_01, dense_ly, dropout_02):
    
    X_train, y_train, X_valid, y_valid, X_test, y_test = split_datasets(r, X, delay, n_features)
   
    model = convlstm(X_train, y_train, X_valid, y_valid, delay, n_features, ftr_ly_01, ftr_ly_02, krn_ly_01, krn_ly_02, dropout_01, dense_ly, dropout_02)
           
    # reshape into subsequences (samples, time steps, rows, cols, channels)
    verbose, n_steps, n_length = 1, 4,30  #selected after analyzing the process for adjusting the model data input (tensor formation)
        
    X_test = X_test.reshape((X_test.shape[0], n_steps, 1, n_length, n_features))
        
    # predict:
    y_pred = np.zeros(y_test.shape)
    id_pred = model.predict_classes(X_test,verbose=verbose)
    y_pred[id_pred==1,1]=1
    y_pred[id_pred==0,0]=1
    
    accuracy_t, precision_t, recall_t, f1_t, MCC_t = evaluate_model(y_test, y_pred)    

    
    return accuracy_t, precision_t, recall_t, f1_t, MCC_t, model, X_test, y_test


def split_datasets(r, X, delay, n_features):
    
    yt = X[:, delay * n_features:]

    # Separate majority and minority classes (to find out the imbalance)
    X_majority = X[yt[:, 0] == 0]
    X_minority = X[yt[:, 0] == 1]

    del X

    # selects 80% of data matrices for training set
    split_index_0 = int(len(X_majority) * 0.8)
    split_index_1 = int(len(X_minority) * 0.8)

    X_tr_0, X_tst_0 = X_majority[:split_index_0], X_majority[split_index_0:]
    X_tr_1, X_tst_1 = X_minority[:split_index_1], X_minority[split_index_1:]

    # Undersampling majority class: To keep the training set data balanced
    X_tr_0 = X_tr_0[:len(X_tr_1)]

    # concatenate the input dataset
    X_train = np.concatenate([X_tr_0, X_tr_1])

    # TRAINING DATASET: defines input and output of training dataset
    y_train = X_train[:, delay * n_features:]
    X_train = X_train[:, :delay * n_features]

    # Total samples for test and validation will be: N_tst_val (corresponding to 80% of the total training set data)
    N_tr = X_train.shape[0]
    N_tst_val = int(N_tr * 100 / 80 - N_tr)

    # KEEP VALIDATION AND TEST SET IN THE ORIGINAL PROPORTION
    X_tst_0 = X_tst_0[:int(N_tst_val * 0.93)]
    X_tst_1 = X_tst_1[:int(N_tst_val * 0.07)]

    split_index_tst_0 = len(X_tst_0) // 2
    split_index_tst_1 = len(X_tst_1) // 2

    X_val_0, X_tst_0 = X_tst_0[:split_index_tst_0], X_tst_0[split_index_tst_0:]
    X_val_1, X_tst_1 = X_tst_1[:split_index_tst_1], X_tst_1[split_index_tst_1:]

    # concatenate the validation and test dataset
    X_valid = np.concatenate([X_val_0, X_val_1])
    X_test = np.concatenate([X_tst_0, X_tst_1])

    # defines input and output of validation and test datasets
    y_valid = X_valid[:, delay * n_features:]
    X_valid = X_valid[:, :delay * n_features]

    y_test = X_test[:, delay * n_features:]
    X_test = X_test[:, :delay * n_features]

    # Z-SCORE
    norm_zscore = StandardScaler()
    X_train = norm_zscore.fit_transform(X_train)
    X_valid = norm_zscore.transform(X_valid)
    X_test = norm_zscore.transform(X_test)

    # adjustment of the initial formation of the input matrices
    X_train = X_train.reshape(X_train.shape[0], delay, n_features)
    X_valid = X_valid.reshape(X_valid.shape[0], delay, n_features)
    X_test = X_test.reshape(X_test.shape[0], delay, n_features)

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def split_dataset(X, delay, n_features):
    # used only for training and testing the last neural network (to be used in the heuristic, selected r=1)
    r = 1 
            
        yt = X[:, delay * n_features:]

    # Separate majority and minority classes (to find out the imbalance)
    X_majority = X[yt[:, 0] == 0]
    X_minority = X[yt[:, 0] == 1]

    del X

    # selects 80% of data matrices for training set
    split_index_0 = int(len(X_majority) * 0.8)
    split_index_1 = int(len(X_minority) * 0.8)

    X_tr_0, X_tst_0 = X_majority[:split_index_0], X_majority[split_index_0:]
    X_tr_1, X_tst_1 = X_minority[:split_index_1], X_minority[split_index_1:]

    # Undersampling majority class: To keep the training set data balanced
    X_tr_0 = X_tr_0[:len(X_tr_1)]

    # concatenate the input dataset
    X_train = np.concatenate([X_tr_0, X_tr_1])

    # TRAINING DATASET: defines input and output of training dataset
    y_train = X_train[:, delay * n_features:]
    X_train = X_train[:, :delay * n_features]

    # Total samples for test and validation will be: N_tst_val (corresponding to 80% of the total training set data)
    N_tr = X_train.shape[0]
    N_tst_val = int(N_tr * 100 / 80 - N_tr)

    # KEEP VALIDATION AND TEST SET IN THE ORIGINAL PROPORTION
    X_tst_0 = X_tst_0[:int(N_tst_val * 0.93)]
    X_tst_1 = X_tst_1[:int(N_tst_val * 0.07)]

    split_index_tst_0 = len(X_tst_0) // 2
    split_index_tst_1 = len(X_tst_1) // 2

    X_val_0, X_tst_0 = X_tst_0[:split_index_tst_0], X_tst_0[split_index_tst_0:]
    X_val_1, X_tst_1 = X_tst_1[:split_index_tst_1], X_tst_1[split_index_tst_1:]

    # concatenate the validation and test dataset
    X_valid = np.concatenate([X_val_0, X_val_1])
    X_test = np.concatenate([X_tst_0, X_tst_1])

    # defines input and output of validation and test datasets
    y_valid = X_valid[:, delay * n_features:]
    X_valid = X_valid[:, :delay * n_features]

    y_test = X_test[:, delay * n_features:]
    X_test = X_test[:, :delay * n_features]

    # Z-SCORE
    norm_zscore = StandardScaler()
    X_train = norm_zscore.fit_transform(X_train)
    X_valid = norm_zscore.transform(X_valid)
    X_test = norm_zscore.transform(X_test)

    # adjustment of the initial formation of the input matrices
    X_train = X_train.reshape(X_train.shape[0], delay, n_features)
    X_valid = X_valid.reshape(X_valid.shape[0], delay, n_features)
    X_test = X_test.reshape(X_test.shape[0], delay, n_features)

    return X_train, y_train, X_valid, y_valid, X_test, y_test


# fit and evaluate a model
def convlstm(X_train, y_train, X_valid, y_valid, delay, n_features, ftr_ly_01, ftr_ly_02, krn_ly_01, krn_ly_02, dropout_01, dense_ly, dropout_02):
    
    # training parameters
    verbose, epochs, batch_size = 0, 200, 1200
    
    # from the data collects the number of input variables (i.e., 4) and output size.
    n_outputs = y_train.shape[1]
    
    # reshape into subsequences (samples, time steps, rows, cols, channels)
    n_steps, n_length = 4,30  #selected after analyzing the process for adjusting the model data input (tensor formation)
    
    # reshape training, validation, and testing sets
    X_train = X_train.reshape((X_train.shape[0], n_steps, 1, n_length, n_features))
    X_valid = X_valid.reshape((X_valid.shape[0], n_steps, 1, n_length, n_features))   
    
    # loss and metrics
    my_loss = tf.keras.losses.CategoricalCrossentropy()
    my_metric1 = tf.keras.metrics.CategoricalAccuracy() 
    my_metric2 = tf.keras.metrics.AUC()
        
   
	# define model
    model = Sequential()
    model.add(ConvLSTM2D(filters=ftr_ly_01, kernel_size=(1,krn_ly_01), activation='tanh', return_sequences=True, input_shape=(n_steps, 1, n_length, n_features)))
    model.add(ConvLSTM2D(filters=ftr_ly_02, kernel_size=(1,krn_ly_02), activation='tanh'))
    model.add(Dropout(dropout_01))
    model.add(Flatten())
    model.add(Dense(dense_ly, activation='relu'))
    model.add(Dropout(dropout_02))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss=my_loss, optimizer='adam', 
               metrics=[my_metric1, my_metric2])
    model.name = 'ConvLSTM'
    
    

	# fit network
    callback = tf.keras.callbacks.EarlyStopping(min_delta=0.0005, patience=6)
    
    history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), 
                        epochs=epochs, batch_size=batch_size, verbose=verbose,
                        callbacks=[callback])    
    
    return model


# MODEL WITH HEURISTIC
def convlstm_heuristic(X_test, y_test, model):
       
    # reshape into subsequences (samples, time steps, rows, cols, channels)
    n_steps, n_length, n_features = 4, 30, 4  #selected after analyzing the process for adjusting the model data input (tensor formation)
    
    # reshape training, validation, and testing sets
    X_test = X_test.reshape((X_test.shape[0], n_steps, 1, n_length, n_features))
    
    # sem heuristica:          
    y_pred = np.zeros(y_test.shape)
    id_pred = model.predict_classes(X_test,verbose=1)
    y_pred[id_pred==1,1]=1
    y_pred[id_pred==0,0]=1
    
    y_after_heuristic = heuristic(y_pred, X_test)
    
    accuracy_h, precision_h, recall_h, f1_h, MCC_h = evaluate_model(y_test, y_after_heuristic)  
        
    return accuracy_h, precision_h, recall_h, f1_h, MCC_h


# summarize scores
def summarize_results(accuracy, precision, recall, f1, MCC):

    print('Accuracy \t  \t \t \t \t %.3f%% (+/-%.3f)' % (mean(accuracy), std(accuracy)))   
    print('Precision \t \t \t \t \t %.3f%% (+/-%.3f)' % (mean(precision), std(precision)))
    print('Recall \t \t \t \t \t \t %.3f%% (+/-%.3f)' % (mean(recall), std(recall)))
    print('F1 \t \t \t \t \t \t %.3f%% (+/-%.3f)' % (mean(f1), std(f1)))
    print('MCC \t \t \t \t \t \t %.3f (+/-%.3f)' % (mean(MCC), std(MCC)))
        
        

def summarize_results_point(accuracy, precision, recall, f1, MCC):
    
    print('Accuracy \t  \t \t \t \t %.3f%%' % (accuracy))   
    print('Precision \t \t \t \t \t %.3f%%' % (precision))
    print('Recall \t \t \t \t \t \t %.3f%%' % (recall))
    print('F1 \t \t \t \t \t \t %.3f%%' % (f1))
    print('MCC \t \t \t \t \t \t %.3f' % (MCC))
    
        
def heuristic(y_antes_heuristica, X_test_heuristica):
    
    # defines the window value (janela), min threshold (limiar_min) and maximum threshold (limiar_max)
    janela, limiar_min, limiar_max = 40, 0.3, 0.8
    
    janelamento = list(range(0,y_antes_heuristica.shape[0], janela))                
    y_pOS_heuristica = np.zeros(y_antes_heuristica.shape)
    

    for i in janelamento:
        # mean < limiar_min
        if mean(y_antes_heuristica[i:i+janela,0]) < limiar_min: 
        
            # se mean < limiar_min -> NORMAL        
            y_pOS_heuristica[i:i+janela,0]=0
            y_pOS_heuristica[i:i+janela,1]=1  
            
        else:
            if mean(y_antes_heuristica[i:i+janela,0]) > limiar_max:     
            
                # se mean > limiar_max -> CLOGGING
                y_pOS_heuristica[i:i+janela,0]=1
                y_pOS_heuristica[i:i+janela,1]=0
                
            else:
                if i>0:
                    # if the previous window is NORMAL -> NORMAL
                    if y_pOS_heuristica[i-1,0] == 0:
                        y_pOS_heuristica[i:i+janela,0]=0
                        y_pOS_heuristica[i:i+janela,1]=1
                    else:
                        # if the previous window is CLOGGING -> CLOGGING
                        y_pOS_heuristica[i:i+janela,0]=1
                        y_pOS_heuristica[i:i+janela,1]=0
                else:
                     # the first is NORMAL (the system always starts in normal mode)
                     y_pOS_heuristica[i:i+janela,0]=0
                     y_pOS_heuristica[i:i+janela,1]=1   
                     
    return y_pOS_heuristica
