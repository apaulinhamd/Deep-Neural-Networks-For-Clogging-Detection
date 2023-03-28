# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 11:47:23 2022

@author: Ana Paula
"""

from IPython import get_ipython
get_ipython().magic('reset -sf')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
from sklearn.metrics import mean_absolute_error 
import math 
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
    
# function responsible for generating the 4x120 input matrix
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

# load dataset
df = pd.read_csv('./database.csv')
X = df.values
X = X.astype('float32')
del df

# Time Matrix X_4x120 (the first 4 variables of the data set were used.)
delay = 120 #determines the temporal sample size of each input matrix
Xt = series_to_supervised(X[:,:4], (delay-1), 1)
Xt = Xt.values

# formation of the output variable
yt = X[(delay-1):,4:]

# define number of features and classes
n_features = X[:,:4].shape[1]
n_classes = 1

# concatenate data (input + output)
X = np.concatenate([Xt,yt],axis=1)

# Separate majority and minority classes
X_majority = X[yt[:,0]==0]
X_minority = X[yt[:,0]==1]

del X


# fit and evaluate a model
def evaluate_model(X_train, y_train, X_valid, y_valid, X_test, y_test):
    
    # training parameters
    verbose, epochs, batch_size = 1, 200, 1200
    
    # from the data collects the number of input variables (i.e., 4) and output size.
    n_features, n_outputs = X_train.shape[2], y_train.shape[1]
    
    # reshape into subsequences (samples, time steps, rows, cols, channels)
    n_steps, n_length = 4,30  #selected after analyzing the process for adjusting the model data input (tensor formation)
    
    # reshape training, validation, and testing sets
    X_train = X_train.reshape((X_train.shape[0], n_steps, 1, n_length, n_features))
    X_valid = X_valid.reshape((X_valid.shape[0], n_steps, 1, n_length, n_features))
    X_test = X_test.reshape((X_test.shape[0], n_steps, 1, n_length, n_features))
    
    # loss and metrics
    my_loss = tf.keras.losses.CategoricalCrossentropy()
    my_metric1 = tf.keras.metrics.CategoricalAccuracy() 
    my_metric2 = tf.keras.metrics.AUC()
        
	# define model
    model = Sequential()
    model.add(ConvLSTM2D(filters=128, kernel_size=(1,5), activation='tanh', return_sequences=True, input_shape=(n_steps, 1, n_length, n_features)))
    model.add(ConvLSTM2D(filters=64, kernel_size=(1,3), activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss=my_loss, optimizer='adam', 
               metrics=[my_metric1, my_metric2])
       
    model.summary()
    

	# fit network
    callback = tf.keras.callbacks.EarlyStopping(min_delta=0.0005, patience=6)
    
    # training time calculation
    import time
    start = time.clock() 
    history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), 
                        epochs=epochs, batch_size=batch_size, verbose=verbose,
                        callbacks=[callback])
    end = time.clock() 
    
    time2train = end-start
    
    del start, end
    	
    #plot history
    plt.figure()
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.legend()
    plt.show()
    
    # predict:
    y_pred = np.zeros(y_test.shape)
    start = time.clock()
    id_pred = model.predict_classes(X_test,verbose=verbose)
    end = time.clock() 
    y_pred[id_pred==1,1]=1
    y_pred[id_pred==0,0]=1
    time2test = end-start  #test time calculation

    
    # analysis of the results by calculating the confusion matrix and defined performance criteria.
    target_names = ['Non-Clogging', 'Clogging']
    print(classification_report(y_test[:,0], y_pred[:,0], target_names=target_names))
    AUC_test = roc_auc_score(y_test[:,0], y_pred[:,0])
    tpr, trc, th = precision_recall_curve(y_test[:,0], y_pred[:,0])
    PR_test =  auc(tpr, trc)
    
    del tpr, trc, th
    
    tn, fp, fn, tp = confusion_matrix(y_test[:,0], y_pred[:,0]).ravel()
    accuracy_test = (tp + tn)/(tp + tn + fp + fn)
    precision_test = tp/(tp + fp)
    recall_test = tp/(tp + fn)
    f1_test = 2*tp/(2*tp + fp + fn)
    MCC_test = ((tp*tn)-(fp*fn))/(math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
    mae_test = mean_absolute_error(y_test[:,0], y_pred[:,0])
    epochs_test = len(history.history['loss'])
    
    return id_pred, time2test, time2train, model, y_pred, AUC_test, accuracy_test, precision_test, recall_test, f1_test, MCC_test, mae_test, PR_test, epochs_test


# summarize scores
def summarize_results(auc, accuracy, precision, recall, f1, MCC, mae, PR, time2train, epochs):
    print('Accuracy \t  \t \t \t \t %.3f%% (+/-%.3f)' % (mean(accuracy), std(accuracy)))   
    print('Precision \t \t \t \t \t %.3f%% (+/-%.3f)' % (mean(precision), std(precision)))
    print('Recall \t \t \t \t \t \t %.3f%% (+/-%.3f)' % (mean(recall), std(recall)))
    print('MCC \t \t \t \t \t \t %.3f (+/-%.3f)' % (mean(MCC), std(MCC)))
    print('F1 \t \t \t \t \t \t \t %.3f%% (+/-%.3f)' % (mean(f1), std(f1)))
    print('MAE \t %.3f%% (+/-%.3f)' % (mean(mae), std(mae)))
    print('AUC (ROC) \t \t \t \t \t %.3f (+/-%.3f)' % (mean(auc), std(auc)))
    print('AUC (PR) \t \t \t \t \t %.3f (+/-%.3f)' % (mean(PR), std(PR)))
    print('Time to Train \t \t \t \t \t \t %.3f (+/-%.3f)' % (mean(time2train), std(time2train)))
    print('Epochs \t \t \t \t \t \t %.3f (+/-%.3f)' % (mean(epochs), std(epochs)))
    
    
# defines the number of times the network will be retrained/tested (a kind of k-fold, with k=5)    
repeats = 5


AUC_ = list()
accuracy_ = list()
precision_ = list()
Recall_ = list()
f1_ = list()
MCC_ = list()
mae_ = list()
PR_ = list()
time2train_ = list()
epochs_ = list()


for r in range(repeats):
    
        
    # selects 80% of data matrices for training set
    # necessary to split for each class of data in order to keep the proportion of the training data set at 50% for each class (data balancing)
    X_tr_0, X_tst_0, y_tr_0, y_tst_0 = train_test_split(X_majority,X_majority[:,0],
                                                        test_size=0.2,  random_state = 59418*r, shuffle=True) # 59418: this number was used as a multiplier of "r" to generate our seed.
    del y_tr_0, y_tst_0 
    
    X_tr_1, X_tst_1, y_tr_1, y_tst_1 = train_test_split(X_minority,X_minority[:,0],
                                                        test_size=0.2, random_state = 59418*r, shuffle=True)
    del y_tr_1, y_tst_1 
    
     
    # Undersampling majoritory class: To keep the training set data balanced
    X_tr_0 = resample(X_tr_0,
                 replace=False,     # sample without replacement
                 n_samples=X_tr_1.shape[0],       # to match minority class
                 random_state=594178*r) # reproducible results
    
    # concatenate the input dataset
    X_train = np.concatenate([X_tr_0, X_tr_1])
    # mixes normal operation and clogging data
    np.random.shuffle(X_train)
    
    # TRAINING DATA SET: defines input and output of training dataset
    y_train = X_train[:,delay*n_features:]
    X_train = X_train[:,:delay*n_features]
        
    
    # Total samples for test and validation will be: N_tst_val (corresponding to 80% of the total training set data)
    N_tr = X_train.shape[0]
    N_tst_val = int(N_tr*100/80 - N_tr)
    
    
    # KEEP VALIDATION AND TEST SET IN THE ORIGINAL PROPORTION
    # resample of validation and test sets (maintaining the original proportion of the distribution of classes (unbalanced set))
    X_tst_0 = resample(X_tst_0,
                 replace=False,    
                 n_samples=int(N_tst_val*0.93), 
                 random_state=594178*r)
    
    X_tst_1 = resample(X_tst_1,
                 replace=False,    
                 n_samples=int(N_tst_val*0.07), 
                 random_state=594178*r)
    
    # half of this set will revert to testing and the other half to validation. This division is made for each class.
    X_tst_0, X_val_0, y_tst_0, y_val_0 = train_test_split(X_tst_0,X_tst_0[:,0],
                                                        test_size=0.5, random_state = 59418*r, shuffle=True)
    del y_tst_0, y_val_0
    
    X_tst_1, X_val_1, y_tst_1, y_val_1 = train_test_split(X_tst_1,X_tst_1[:,0],
                                                        test_size=0.5, random_state = 59418*r, shuffle=True)
    del y_tst_1, y_val_1 
    
    
    # concatenate the validation and test dataset
    X_valid = np.concatenate([X_val_0, X_val_1])
    X_test = np.concatenate([X_tst_0, X_tst_1])    
    
    # mixes normal operation and clogging data
    np.random.shuffle(X_valid)
    np.random.shuffle(X_test)
    
    
    # defines input and output of validation and test datasets
    y_valid = X_valid[:,delay*n_features:]
    X_valid = X_valid[:,:delay*n_features]
        
    y_test = X_test[:,delay*n_features:]
    X_test = X_test[:,:delay*n_features]
        
        
        
    del X_tr_1, X_tst_1
    del X_tr_0, X_tst_0
    del X_val_0, X_val_1
        
    
    # Z-SCORE
    norm_zscore = StandardScaler()
    X_train = norm_zscore.fit_transform(X_train)
    X_valid = norm_zscore.transform(X_valid)
    X_test = norm_zscore.transform(X_test)
    
    # adjustment of the initial formation of the input matrices    
    X_train = X_train.reshape(X_train.shape[0], delay, n_features)
    X_valid = X_valid.reshape(X_valid.shape[0], delay, n_features)
    X_test = X_test.reshape(X_test.shape[0], delay, n_features)
    
    # calls the function responsible for training, validating and testing the neural network
    id_pred, time2test, time2train, model, y_pred, AUC_test, accuracy_test, precision_test, recall_test, f1_test, MCC_test, mae_test, PR_test, epochs_test = evaluate_model(X_train, y_train, X_valid, y_valid, X_test, y_test)       
    
    
    AUC_.append(AUC_test)
    accuracy_.append(accuracy_test*100.0)
    precision_.append(precision_test*100.0)
    Recall_.append(recall_test*100.0)
    f1_.append(f1_test*100.0)
    MCC_.append(MCC_test)
    mae_.append(mae_test*100.0)
    PR_.append(PR_test)
    time2train_.append(time2train)
    epochs_.append(epochs_test)
    
    
    if r != repeats-1:
        del  X_train, X_test, y_train, y_test, X_valid, y_valid, model
        
        
# summarize results
model.summary()
summarize_results(AUC_, accuracy_, precision_, Recall_, f1_, MCC_, mae_, PR_,time2train_,epochs_)





