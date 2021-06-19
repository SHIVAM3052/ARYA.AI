# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 13:37:43 2021

@author: katsh
"""

#import files
import pandas as pd
import json
from keras.models import load_model
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, classification_report

CONFIG_PATH = 'C:/Users/katsh/OneDrive/Desktop/Arya.ai/CODE/config/inference_config.json'

if __name__ == '__main__':
    
    #Reading config_file
    with open(CONFIG_PATH) as f:
        eval_config = json.load(f)
        
    train_dir = eval_config['train_path']
    test_dir = eval_config['test_path']
    model_dir = eval_config['save_model_path']
    
        
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

    # load weights into new model
    model = load_model(model_dir)
    print("Loaded model from disk")

    # evaluate the keras model
    _, accuracy = model.evaluate(X_test, y_test)
    print('#'*100)
    print('Accuracy: %.2f' % (accuracy*100))

    # Predict the keras model
    y_predict = model.predict_classes(X_test)
    cm = confusion_matrix(y_test.to_numpy(), y_predict)
    print('#'*100)
    print('Confusion Matrix')
    print( cm)
    print('#'*100)
    print(classification_report(y_test, y_predict))  
    print('#'*100)
    print('#'*100)
    print('#'*100)

    # Reading testing file
    test_df = pd.read_csv(test_dir)
    df_test = test_df.drop(['Unnamed: 0'], axis=1)

    # Preprocessing Test file
    z = df_test
    scaler = preprocessing.StandardScaler().fit(z)
    z_scaled = scaler.transform(z)
    test_df1 = pd.DataFrame(z_scaled)
    test_df1

    # Predict Y column values on Test dataset
    yhat = model.predict_classes(test_df1)
    print(yhat.shape)
    print('Predicted: %s' % (yhat))
