# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 07:50:17 2022

@author: Ana Paula
"""


from IPython import get_ipython
get_ipython().magic('reset -sf')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import concat
from pandas import DataFrame
from sklearn.utils import resample
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import ConvLSTM2D
from numpy import mean
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
delay = 120 # Time Matrix X_4x120 (the first 4 variables of the data set were used.)
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
start_idx = 0

# Calcula o tamanho dos conjuntos
N_tr = 324634 # 80% das amostras
N_tr_maj = int(0.5 * N_tr) # 50% das amostras = normal
N_tr_min = int(0.5 * N_tr) # 50% das amostras = clogging

N_val = 40579 # 10% das amostras
N_val_maj = int(0.9334 * N_val) # 93,34% das amostras = normal
N_val_min = int(0.0666 * N_val) # 6,66% das amostras = clogging

N_tst = 40579 # 10% das amostras
N_tst_maj = int(0.9334 * N_tst) # 93,34% das amostras = normal
N_tst_min = int(0.0666 * N_tst) # 6,66% das amostras = clogging


# Amostras de treinamento
end_train_maj = start_idx + N_tr_maj
end_train_min = start_idx + N_tr_min
X_tr_0 = X_majority[start_idx:end_train_maj]
X_tr_1 = X_minority[start_idx:end_train_min]

# Amostras de validação
start_val = end_train_maj + 1
end_val_maj = start_val + N_val_maj
end_val_min = start_val + N_val_min
X_val_0 = X_majority[start_val:end_val_maj]
X_val_1 = X_minority[start_val:end_val_min]

# Amostras de teste
start_test = end_val_maj + 1
end_test_maj = start_test + N_tst_maj
end_test_min = start_test + N_tst_min
X_tst_0 = X_majority[start_test:end_test_maj]
X_tst_1 = X_minority[start_test:end_test_min]


# concatenate the input dataset
X_train = np.concatenate([X_tr_0, X_tr_1])
np.random.seed(2763)
np.random.shuffle(X_train)

# TRAINING DATASET: defines input and output of training dataset
y_train = X_train[:, delay * n_features:]
X_train = X_train[:, :delay * n_features]

# concatenate the validation and test dataset
X_valid = np.concatenate([X_val_0, X_val_1])
X_test = np.concatenate([X_tst_0, X_tst_1])   
np.random.shuffle(X_valid)
np.random.shuffle(X_test)

# defines input and output of validation and test datasets
y_valid = X_valid[:, delay * n_features:]
X_valid = X_valid[:, :delay * n_features]

y_test = X_test[:, delay * n_features:]
X_test = X_test[:, :delay * n_features]

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
    

# classification before heuristic

# training parameters
verbose, epochs, batch_size = 1, 200, 1200

# from the data collects the number of input variables (i.e., 4) and output size.
n_features, n_outputs = X_train.shape[2], y_train.shape[1]

# reshape into subsequences (samples, time steps, rows, cols, channels)
n_steps, n_length = 4,30#selected after analyzing the process for adjusting the model data input (tensor formation)
    
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
model.add(ConvLSTM2D(filters=256, kernel_size=(1,5), activation='tanh', return_sequences=True, input_shape=(n_steps, 1, n_length, n_features)))
model.add(ConvLSTM2D(filters=16, kernel_size=(1,3), activation='tanh'))
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
time2test = end-start


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
    
# summarize scores
print('Accuracy \t  \t \t \t \t %.3f%%' % (accuracy_test*100.0))  
print('Precision \t \t \t \t \t %.3f%%' % (precision_test*100.0))
print('Recall \t \t \t \t \t \t %.3f%%' % (recall_test*100.0))
print('MCC \t \t \t \t \t \t %.3f' % (MCC_test))
print('F1 \t \t \t \t \t \t \t %.3f%%' % (f1_test*100.0))
print('MAE \t %.3f%%' % (mae_test))
print('AUC (ROC) \t \t \t \t \t %.3f' % (AUC_test))
print('AUC (PR) \t \t \t \t \t %.3f' % (PR_test))
print('Time to Train \t \t \t \t \t \t %.3f' % (time2train))
print('Epochs \t \t \t \t \t \t %.3f' % (epochs_test))
      

# heuristic
window = 40 # sets the size of the evaluation window
windowing = list(range(0,len(X_test), window)) #

y_before_heuristic = y_pred
y_pOS_heuristic = np.zeros(y_pred.shape)


#s et the max and min limit values
threshold_min = 0.3
threshold_max = 0.8

for i in windowing:
    if mean(y_before_heuristic[i:i+window ,0]) < threshold_min: 
    
        # if mean < threshold min -> NORMAL        
        y_pOS_heuristic[i:i+window ,0]=0
        y_pOS_heuristic[i:i+window ,1]=1  
        
    else:
        if mean(y_before_heuristic[i:i+window ,0]) > threshold_max:     
        
            # if mean > threshold max -> CLOGGING
            y_pOS_heuristic[i:i+window ,0]=1
            y_pOS_heuristic[i:i+window ,1]=0
            
        else:
            if i>0:
                # if the previous window is NORMAL -> NORMAL
                if y_pOS_heuristic[i-1,0] == 0:
                    y_pOS_heuristic[i:i+window ,0]=0
                    y_pOS_heuristic[i:i+window ,1]=1
                else:
                    # if the previous window is CLOGGING -> CLOGGING
                    y_pOS_heuristic[i:i+window ,0]=1
                    y_pOS_heuristic[i:i+window ,1]=0
            else:
                 # the first is NORMAL (the system always starts with normal operation)
                 y_pOS_heuristic[i:i+window ,0]=0
                 y_pOS_heuristic[i:i+window ,1]=1   
        

# analysis of the results calculated by the confusion matrix after applying the heuristic
AUC_test_pOS_heur = roc_auc_score(y_test[:,0], y_pOS_heuristic[:,0])

tn_pOS, fp_pOS, fn_pOS, tp_pOS = confusion_matrix(y_test[:,0], y_pOS_heuristic[:,0]).ravel()
accuracy_test_pOS_heur = (tp_pOS + tn_pOS)/(tp_pOS + tn_pOS + fp_pOS + fn_pOS)
precision_test_pOS_heur = tp_pOS/(tp_pOS + fp_pOS)
recall_test_pOS_heur = tp_pOS/(tp_pOS + fn_pOS)
f1_test_pOS_heur = 2*tp_pOS/(2*tp_pOS + fp_pOS + fn_pOS)
MCC_test_pOS_heur = ((tp_pOS*tn_pOS)-(fp_pOS*fn_pOS))/(math.sqrt((tp_pOS+fp_pOS)*(tp_pOS+fn_pOS)*(tn_pOS+fp_pOS)*(tn_pOS+fn_pOS)))

mae_test_pOS_heur = mean_absolute_error(y_test[:,0], y_pOS_heuristic[:,0])

# summarize new scores
print('Accuracy (heuristic) \t  \t \t \t \t %.3f%%' % (accuracy_test_pOS_heur*100.0))  
print('Precision (heuristic) \t \t \t \t \t %.3f%%' % (precision_test_pOS_heur*100.0))
print('Recall (heuristic) \t \t \t \t \t \t %.3f%%' % (recall_test_pOS_heur*100.0))
print('MCC (heuristic) \t \t \t \t \t \t %.3f' % (MCC_test_pOS_heur))
print('F1 (heuristic) \t \t \t \t \t \t \t %.3f%%' % (f1_test_pOS_heur*100.0))
print('MAE (heuristic) \t %.3f%%' % (mae_test_pOS_heur))
print('AUC (ROC) (heuristic) \t \t \t \t \t %.3f' % (AUC_test_pOS_heur))

    


