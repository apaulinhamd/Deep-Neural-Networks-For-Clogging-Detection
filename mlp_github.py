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
from keras.layers import Dropout
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



# convert series to supervised learning
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

# Time Matrix X_4x120
delay = 120
Xt = series_to_supervised(X[:,:4], (delay-1), 1)
Xt = Xt.values

# output
yt = X[(delay-1):,4:]

# features and classes
n_features = X[:,:4].shape[1]
n_classes = 1

# concatenate data
X = np.concatenate([Xt,yt],axis=1)


# fit and evaluate a model
def evaluate_model(X_train, y_train, X_valid, y_valid, X_test, y_test):
    
    verbose, epochs, batch_size = 1, 200, 1200
    n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], y_train.shape[1]
 
    X_train = X_train.reshape(X_train.shape[0], n_timesteps*n_features, order = 'F')
    X_valid = X_valid.reshape(X_valid.shape[0],  n_timesteps*n_features, order = 'F')
    X_test = X_test.reshape(X_test.shape[0],  n_timesteps*n_features, order = 'F')
  
    
    my_loss = tf.keras.losses.CategoricalCrossentropy()
    my_metric1 = tf.keras.metrics.CategoricalAccuracy() #tf.keras.metrics.Accuracy()
    my_metric2 = tf.keras.metrics.AUC()
        
    input_shape = (n_timesteps*n_features,)
        
	# define model1w
    model = Sequential()
    model.add(Dense(64, input_shape = input_shape, activation='sigmoid'))
    #model.add(Dropout(0.3))
    model.add(Dense(32, activation='sigmoid'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss=my_loss, optimizer='adam', 
               metrics=[my_metric1, my_metric2])

    model.summary()
    

	# fit network
    callback = tf.keras.callbacks.EarlyStopping(min_delta=0.0005, patience=6)
    
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
    
    
    y_pred = np.zeros(y_test.shape)
    start = time.clock()
    id_pred = model.predict_classes(X_test,verbose=verbose)
    end = time.clock() 
    y_pred[id_pred==1,1]=1
    y_pred[id_pred==0,0]=1
    time2test = end-start

    
    
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

# identifica as posições na matriz one é clogging (1) e operação normal (0), salvando as posições
posicoes_0 = np.where(yt[:,0]==0)[0]
posicoes_1 = np.where(yt[:,0]==1)[0]

# Calcula o tamanho dos conjuntos
N_tr = 324634 # 80% das amostras
N_tr_0 = int(0.5 * N_tr) # 50% das amostras = normal
N_tr_1 = int(0.5 * N_tr) # 50% das amostras = clogging

N_val = 40579 # 10% das amostras
N_val_0 = int(0.9334 * N_val) # 93,34% das amostras = normal
N_val_1 = int(0.0666 * N_val) # 6,66% das amostras = clogging

N_tst = 40579 # 10% das amostras
N_tst_0 = int(0.9334 * N_tst) # 93,34% das amostras = normal
N_tst_1 = int(0.0666 * N_tst) # 6,66% das amostras = clogging


def gerar_datasets(N_tst, N_val, N_tr, repeats, N_tr_0, N_tr_1, N_val_0, N_val_1, N_tst_0, N_tst_1, posicoes_0, posicoes_1):
    posicoes = []

    for r in range(repeats):
        
        if r == 0:
            # Atualiza o índice de início para o próximo fold 
            start_idx_0 = 0
            start_idx_1 = 1
        
        # Amostras de treinamento
        end_train_0 = start_idx_0 + N_tr_0
        end_train_1 = start_idx_1 + N_tr_1
        X_tr_0 = posicoes_0[start_idx_0:end_train_0]
        X_tr_1 = posicoes_1[start_idx_1:end_train_1]
        
        # Identificando qual é a maior posição entre X_tr_1 e X_tr_0
        #maior_posicao = max(X_tr_1[-1] if X_tr_1 else 0, X_tr_0[-1] if X_tr_0 else 0)
        maior_posicao_1 = X_tr_1[-1] if X_tr_1.any() else 0
        maior_posicao_0 = X_tr_0[-1] if X_tr_0.any() else 0
        maior_posicao = max(maior_posicao_1, maior_posicao_0)
        
        del maior_posicao_0, maior_posicao_1

        # Inicializando a próxima posição
        start_val = maior_posicao + 1
    
        if start_val in posicoes_0:
            vstart_val_0 = start_val
            vstart_val_1 = min(posicoes_1, key=lambda x: abs(x - start_val))
            
        else:
             vstart_val_1 = start_val
             vstart_val_0 = min(posicoes_0, key=lambda x: abs(x - start_val))
               
        start_val_1 = int(np.where(posicoes_1 == vstart_val_1)[0])
        start_val_0 = int(np.where(posicoes_0 == vstart_val_0)[0])
    
        # Amostras de validação    
        end_val_0 = start_val_0 + N_val_0
        end_val_1 = start_val_1 + N_val_1
        X_val_0 = posicoes_0[start_val_0:end_val_0]
        X_val_1 = posicoes_1[start_val_1:end_val_1]
    
    
        # Identificando qual é a maior posição entre X_tr_1 e X_tr_0       
        maior_posicao_1 = X_val_1[-1] if X_val_1.any() else 0
        maior_posicao_0 = X_val_0[-1] if X_val_0.any() else 0
        maior_posicao = max(maior_posicao_1, maior_posicao_0)
        
        del maior_posicao_0, maior_posicao_1
        
        
        # Inicializando a próxima posição
        start_tst = maior_posicao + 1
    
        if start_tst in posicoes_0:
            vstart_tst_0 = start_tst
            vstart_tst_1 = min(posicoes_1, key=lambda x: abs(x - start_tst))
            
        else:
             vstart_tst_1 = start_tst
             vstart_tst_0 = min(posicoes_0, key=lambda x: abs(x - start_tst))
           
        start_tst_1 = int(np.where(posicoes_1 == vstart_tst_1)[0])
        start_tst_0 = int(np.where(posicoes_0 == vstart_tst_0)[0])
    
       
        # Amostras de teste    
        end_tst_0 = start_tst_0 + N_tst_0
        end_tst_1 = start_tst_1 + N_tst_1
        X_tst_0 = posicoes_0[start_tst_0:end_tst_0]
        X_tst_1 = posicoes_1[start_tst_1:end_tst_1]
        
        #Identificando qual é a maior posição entre X_tr_1 e X_tr_0
        
        maior_posicao_1 = X_tst_1[-1] if X_tst_1.any() else 0
        maior_posicao_0 = X_tst_0[-1] if X_tst_0.any() else 0
        maior_posicao = max(maior_posicao_1, maior_posicao_0)
        
        del maior_posicao_0, maior_posicao_1
        
        # Inicializando a próxima posição
        start_idx = maior_posicao + 1
    
        if start_idx in posicoes_0:
            vstart_idx_0 = start_idx
            vstart_idx_1 = min(posicoes_1, key=lambda x: abs(x - start_idx))
            
        else:
             vstart_idx_1 = start_idx
             vstart_idx_0 = min(posicoes_0, key=lambda x: abs(x - start_idx))
           
        #start_idx_1 = posicoes_1.index(vstart_idx_1)   
        #start_idx_0 = posicoes_0.index(vstart_idx_0)  

        start_idx_1 = int(np.where(posicoes_1 == vstart_idx_1)[0])
        start_idx_0 = int(np.where(posicoes_0 == vstart_idx_0)[0])
    
        var_tr = np.concatenate([X_tr_0, X_tr_1])
        var_tr = np.sort(var_tr)

        var_val = np.concatenate([X_val_0, X_val_1])
        var_val = np.sort(var_val)
        
        var_tst = np.concatenate([X_tst_0, X_tst_1])
        var_tst = np.sort(var_tst)

        posicoes.append((var_tr, var_val, var_tst))
        
    return posicoes


posicoes = gerar_datasets(N_tst, N_val, N_tr, repeats, N_tr_0, N_tr_1, N_val_0, N_val_1, N_tst_0, N_tst_1, posicoes_0, posicoes_1)


for r in range(repeats):
    posicao_tr = posicoes[r][0]
    posicao_val = posicoes[r][1]
    posicao_tst = posicoes[r][2]
    
    # TRAINING DATASET: defines input and output of training dataset
    X_tr = X[posicao_tr,:]
    y_train = X_tr[:, delay * n_features:]
    X_train = X_tr[:, :delay * n_features]

    # defines input and output of validation and test datasets
    X_val = X[posicao_val,:]
    y_valid = X_val[:, delay * n_features:]
    X_valid = X_val[:, :delay * n_features]

    X_tst = X[posicao_tst,:]
    y_test = X_tst[:, delay * n_features:]
    X_test = X_tst[:, :delay * n_features]

    ##### NORMALIZAÇÃO Z-SCORE ##### 
    
    norm_zscore = StandardScaler()
    
    X_train = norm_zscore.fit_transform(X_train)
    X_valid = norm_zscore.transform(X_valid)
    X_test = norm_zscore.transform(X_test)
    
    
    X_train = X_train.reshape(X_train.shape[0], delay, n_features)
    X_valid = X_valid.reshape(X_valid.shape[0], delay, n_features)
    X_test = X_test.reshape(X_test.shape[0], delay, n_features)
    
    
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


from sklearn.metrics import confusion_matrix
tn, fp, fn, tp = confusion_matrix(y_test[:,0], y_pred[:,0]).ravel()

fa = 100*fp/(tn+fp)




