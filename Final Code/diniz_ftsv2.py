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

    start_idx = r * (N_tr + N_val + N_tst)

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
    
    # TRAINING DATASET: defines input and output of training dataset
    y_train = X_train[:, delay * n_features:]
    X_train = X_train[:, :delay * n_features]
    
    # concatenate the validation and test dataset
    X_valid = np.concatenate([X_val_0, X_val_1])
    X_test = np.concatenate([X_tst_0, X_tst_1])   

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
    
    y_after_heuristic = heuristic(y_pred)
    
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
    
        
def heuristic(y_before_heuristic, X_test_heuristica):
    
    # defines the window value (janela), min threshold (limiar_min) and maximum threshold (limiar_max)
    janela, limiar_min, limiar_max = 40, 0.3, 0.8
    
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
                        # Se janela anterior for completamente clogging (as 40 amostras) - CLOGGING
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

    return y_pOS_heuristica
