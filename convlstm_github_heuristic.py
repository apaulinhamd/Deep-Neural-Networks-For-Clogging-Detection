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

repeats = 1
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


r = 1
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
#window = 40 # sets the size of the evaluation window
#windowing = list(range(0,len(X_test), window)) #

#y_before_heuristic = y_pred
#y_pOS_heuristic = np.zeros(y_pred.shape)


#max and min limit values
#threshold_min = 0.3
#threshold_max = 0.8

# defines the window value (janela), min threshold (limiar_min) and maximum threshold (limiar_max)
janela, limiar_min, limiar_max = 45, 0.3, 0.8

y_pOS_heuristica = np.zeros(y_before_heuristic.shape)
janelamento = list(range(0,y_before_heuristic.shape[0]-janela, 1)) 
op1, op2, op3, op4, op5, op6, op7, op8 = 0, 0, 0, 0, 0, 0, 0, 0

# FUTURO y_pOS_heuristica[i,0] - preencho em t=0 com base em 0 a 40
for i in janelamento:
    Clg = y_before_heuristic[i:i+janela,0]
    Clg_index = np.where(Clg == 1)[0]
    
    if i>janela:
        # conta o total de amostras consecutivas = 1 no inicio da janela ATUAL
        j1=0
        while Clg.any():
    	if j1<len(Clg):
    	   if Clg[j1]==1:
    		j1 += 1
    	    else:
    		break
    	else:
    	    break       
    
        # conta o total de amostras consecutivas = 1 no final da janela ANTERIOR   
        Clg_anterior = y_pOS_heuristica[i-janela:i,0]
        
        jj=len(Clg_anterior)-1 
        
        while Clg_anterior.any():
           if jj != 0:
    	   if Clg_anterior[jj]==1:
    	       jj -= 1
    	   else:
    	       break
           else:
    	   break   
        j2 = (len(Clg_anterior)-1)-jj
        j = j2+j1
    		       
        # media < limiar_min
        if mean(Clg) <= limiar_min: # COMO 0.3 = 13.5 
    	       
    	# se a amostra atual for clogging 
    	# E a janela anterior for toda clogging - CLOGGING
    	if y_before_heuristic[i,0]==1 and j >= 45:
    	    y_pOS_heuristica[i,0] = 1
    	    y_pOS_heuristica[i,1] = 0
    	    op1+=1
    	    
    	# caso contrário, será NORMAL
    	else: 
    	    y_pOS_heuristica[i,0] = 0
    	    y_pOS_heuristica[i,1] = 1
    	    op2+=1
        
        else:
    	# se media > lim_max    
    	if mean(Clg) >= limiar_max: # COM 0.8 = 36
    		y_pOS_heuristica[i,0] = 1
    		y_pOS_heuristica[i,1] = 0   
    		op3+=1
    	
    	# se media > lim_min ou media < lim_max
    	else:            
    	    # se tiver amostra 1 na extremidade do inicio - CLOGGING
    	    if Clg_index[0] == 0:
    		
    		# Se a qnt de amostras somadas (janela atual e anterior) é >= 45 (é indicativo de q ainda existe CLOGGING, então não altero essas variáveis)
    		if j >= 45:
    		    y_pOS_heuristica[i,0] = 1
    		    y_pOS_heuristica[i,1] = 0
    		    op4+=1
    
    		# Se a qnt de amostras somadas é < 45 (é indicativo de op. NORMAL)
    		else:
    		    y_pOS_heuristica[i,0] = 0
    		    y_pOS_heuristica[i,1] = 1
    		    op5+=1
    	    
    	    # se amostra da extremidade for igual a 0            
    	    else:
    		# Se janela anterior for completamente clogging (as 45 amostras) - CLOGGING
    		if j2 == janela-1:
    		    y_pOS_heuristica[i,0] = 1
    		    y_pOS_heuristica[i,1] = 0
    		    op6+=1
    		
    		# Se for normal > AVALIA AMOSTRA POR AMOSTRA
    		else:  
    		    y_pOS_heuristica[i,0] = 0
    		    y_pOS_heuristica[i,1] = 1
    		    op7+=1
    
    
    else:
         y_pOS_heuristica[i,0] = 0
         y_pOS_heuristica[i,1] = 1 
         op8+=1         
            

# analysis of the results calculated by the confusion matrix after applying the heuristic
AUC_test_pOS_heur = roc_auc_score(y_test[:,0], y_pOS_heuristica[:,0])

tn_pOS, fp_pOS, fn_pOS, tp_pOS = confusion_matrix(y_test[:,0], y_pOS_heuristica[:,0]).ravel()
accuracy_test_pOS_heur = (tp_pOS + tn_pOS)/(tp_pOS + tn_pOS + fp_pOS + fn_pOS)
precision_test_pOS_heur = tp_pOS/(tp_pOS + fp_pOS)
recall_test_pOS_heur = tp_pOS/(tp_pOS + fn_pOS)
f1_test_pOS_heur = 2*tp_pOS/(2*tp_pOS + fp_pOS + fn_pOS)
MCC_test_pOS_heur = ((tp_pOS*tn_pOS)-(fp_pOS*fn_pOS))/(math.sqrt((tp_pOS+fp_pOS)*(tp_pOS+fn_pOS)*(tn_pOS+fp_pOS)*(tn_pOS+fn_pOS)))

mae_test_pOS_heur = mean_absolute_error(y_test[:,0], y_pOS_heuristica[:,0])

# summarize new scores
print('Accuracy (heuristic) \t  \t \t \t \t %.3f%%' % (accuracy_test_pOS_heur*100.0))  
print('Precision (heuristic) \t \t \t \t \t %.3f%%' % (precision_test_pOS_heur*100.0))
print('Recall (heuristic) \t \t \t \t \t \t %.3f%%' % (recall_test_pOS_heur*100.0))
print('MCC (heuristic) \t \t \t \t \t \t %.3f' % (MCC_test_pOS_heur))
print('F1 (heuristic) \t \t \t \t \t \t \t %.3f%%' % (f1_test_pOS_heur*100.0))
print('MAE (heuristic) \t %.3f%%' % (mae_test_pOS_heur))
print('AUC (ROC) (heuristic) \t \t \t \t \t %.3f' % (AUC_test_pOS_heur))

    


