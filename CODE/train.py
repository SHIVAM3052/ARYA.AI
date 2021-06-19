# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 12:18:02 2021

@author: katsh
"""

#import files
import json
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix, classification_report


CONFIG_PATH = 'C:/Users/katsh/OneDrive/Desktop/Arya.ai/CODE/config/train_config.json'

if __name__ == '__main__':
    
    #Reading config_file
    with open(CONFIG_PATH) as f:
        model_config = json.load(f)
        
    train_dir = model_config['train_path']
    test_dir = model_config['test_path']
    model_dir = model_config['save_model_path']
    ckpt_dir = model_config['checkpoint_path']
    
    # Reading training file
    train_df = pd.read_csv(train_dir)
    df_train = train_df.drop(['Unnamed: 0'], axis=1)

    # y includes our labels and x includes our features  
    y = df_train.Y
    x = df_train.drop(['Y'], axis=1)

    # Preprocessing
    scaler = preprocessing.StandardScaler().fit(x)
    X_scaled = scaler.transform(x)
    train_df1 = pd.DataFrame(X_scaled)

    #split train test data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(train_df1, y, random_state=42)

#############################################################################
#Simple ANN model for Binary Classification
    print('Simple ANN model for Binary Classification')
    model = Sequential()
    model.add(Dense(60, input_dim=57, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    checkpoint_filepath = ckpt_dir
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)  

# Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=150,validation_split=0.05, callbacks=[model_checkpoint_callback])

# save model to HDF5
    model.save(model_dir)
    print("Saved model to disk")
    print('#'*100)
    print('#'*100)
    print('#'*100)
    print('#'*100)
    
###################################################################

# Simple ANN model with StratifiedKfold 4:1 for Binary Classification
    print('Simple ANN model with StratifiedKfold for Binary Classification')
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=47)    #COPY 
    cvscores = []
    for train, test in kfold.split(train_df, y):
        # create model
        model1 = Sequential()
        model1.add(Dense(60, input_dim=57, activation='relu'))
        model1.add(Dense(1, activation='sigmoid'))
        # Compile model
        model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # Fit the model
        model1.fit(X_train, y_train, epochs=50, batch_size=10, verbose=0)
        # evaluate the model
        scores = model1.evaluate(X_test, y_test, verbose=0)
        print("%s: %.2f%%" % (model1.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)
    print("Baseline: %.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

################################################################

