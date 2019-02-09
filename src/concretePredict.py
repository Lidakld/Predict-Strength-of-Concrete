# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import sys
import pandas as pd
from pandas.plotting import scatter_matrix
import seaborn as sns
import os
from sklearn.utils import shuffle
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

def line(): print("-----------------------------------\n")
def help():
    print("______\n"+"Usage:\n"+
          "python concretePredict.py <methods_seq> <model seq>")
    line()
    print("methods_name           \t methods_seq\n"+
          "Linear Regression      \t 0\n"+
          "MFNN                   \t 1\n"+
          "Kenerl Ridge Regression\t 2")
    line()
    print("MFNN_model has 0 to 3 model_seq\n"+
          "The other two do not need model_seq");
    line()
    print("Example:\n"+
          "python concretePredict.py 1")
    
def readData():
    try:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        eles = dir_path.split(os.sep)
        eles = eles[0:len(eles)-1]
        csv_path = os.sep.join(eles) + os.sep +"data"+os.sep+'Concrete_Data.csv'
        column_names = ['CEM', 'BLA', 'ASH', 'WAT', 'PLA', 'CAGG', 'FAGG', 'AGE', 'STRENTH']
        concreate_strength = pd.read_csv(csv_path, skiprows=1, names=column_names)
        concreate_strength = shuffle(concreate_strength)
    except:
        line()
        print("Place data file in" +csv_path)
        line()
    return(concreate_strength)
    
def preprocessing(concreate_strength):
    datax = concreate_strength.iloc[:,0:(len(concreate_strength.columns)-1)]
    datay = concreate_strength.iloc[:,(len(concreate_strength.columns)-1):len(concreate_strength.columns)]
                                    
    #normalization to x
    mean = datax.mean(axis=0)
    std = datax.std(axis=0)
    datax = (datax - mean) / std
    
    return(datax, datay)
    
def splite(datax, datay):
    trainx, testx, trainy, testy = train_test_split(datax, datay, test_size=0.2)
    
    return(trainx, testx, trainy, testy)
    
def linear_regression(trainx, trainy,testx, testy):
    lm = linear_model.LinearRegression()

    model = lm.fit(trainx, trainy)
    y_hat = lm.predict(testx)
    
    limMax = max(float(testy.max()), float(y_hat.max()))
    limMin = min(float(testy.min()), float(y_hat.min()))
    plt.plot(testy, y_hat, 'bo')
    plt.xlabel('True Values')
    plt.xlim(limMin, limMax)
    plt.ylabel('Predictions')
    plt.ylim(limMin, limMax)
    plt.savefig('linear_regression.png', dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format='png',
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None)
    plt.show()
    
    print("Lienar regression, Y to Y_hat save to path:" + 'linear_regression.png')
    
    print('Linear regression R2 Score:', model.score(testx, testy))
    
def baseLine(concreate_strength):
    datax, datay = preprocessing(concreate_strength)
    trainx, testx, trainy, testy = splite(datax, datay)
    linear_regression(trainx, trainy,testx, testy)
    
    
def plot_history(history, model_seq):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [1000s]')
    plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']), 
               label='Val loss')
    plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
               label = 'Train Loss')
    plt.legend()
    plt.ylim([0,5])
    plt.savefig('mfnn_loss'+str(model_seq)+'.png', dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format='png',
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None)
    print("\nMFNN<train loss, val loss> save to :" + 'mfnn_loss'+str(model_seq)+'.png')
    
    plt.show()
        
# Display training progress by printing a single dot for each completed epoch.
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs):
        if epoch % 100 == 0: print('')
        print('.', end='')
    
def r2_score(y, y_hat):
    tot = np.sum((y-np.mean(y))**2)
    pre = np.sum((y-y_hat)**2)
    
    return(1 - pre/tot)

def MSE(y,y_hat):
    return (np.sum((y-y_hat)**2))/len(y)

def build_model(train_data, model_seq):
    model_seq = int(model_seq)
#    print("MODEL_SEQ %d" % model_seq)

    if(model_seq == 0):
        model = keras.Sequential([
        keras.layers.Dense(256, activation=tf.nn.relu, 
                            input_shape=(train_data.shape[1],)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense(8, activation=tf.nn.relu),
        keras.layers.Dense(1)
        ])
    elif(model_seq == 1):
        model = keras.Sequential([
            keras.layers.Dense(256, activation=tf.nn.relu, 
                           input_shape=(train_data.shape[1],)),
        keras.layers.Dense(1)
      ])
    elif(model_seq == 2):
         model = keras.Sequential([
        keras.layers.Dense(256, activation=tf.nn.relu, 
                           input_shape=(train_data.shape[1],)),
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense(1)
      ])
    else:
        model = keras.Sequential([
        keras.layers.Dense(30, activation=tf.nn.relu, 
                           input_shape=(train_data.shape[1],)),
        keras.layers.Dense(10, activation=tf.nn.relu),
        keras.layers.Dense(1, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.relu),
        keras.layers.Dense(1)
      ])
        
    optimizer = tf.train.RMSPropOptimizer(0.001)

    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae'])
    return model

def mfnn(concreate_strength, model_seq):
    datax, datay = preprocessing(concreate_strength)
    trainx, testx, trainy, testy = splite(datax, datay)
    print("MFNN Structure:\t\n")
    model = build_model(trainx, model_seq)
    model.summary()
    # The patience parameter is the amount of epochs to check for improvement.
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)
    
    EPOCHS = 500
    
    print("\nTraining Starts:\t\n")
    # Store training stats
    history = model.fit(trainx.copy(), trainy.copy(), epochs=EPOCHS,
                        validation_split=0.2, verbose=0,
                        callbacks=[early_stop, PrintDot()])
    
    plot_history(history, model_seq)
    
    print("\Precition Result:\t\n")
    y_hat = model.predict(testx).flatten()

    r2 = r2_score(testy.values[:,0], y_hat)
    mse = MSE(testy.values[:,0], y_hat)
    print("R2 score: %7.5f" % r2)
    print("MSE: %7.5f" % mse)
    
    
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV

def krr(concreate_strength, model_seq):
    print("______\nKRR mode<"+str(model_seq)+">:")
    line();
    datax, datay = preprocessing(concreate_strength)
    trainx, testx, trainy, testy = splite(datax, datay)
    kr = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1,degree=4), cv=5,
                  param_grid={"alpha": [1, 0.01, 0.1, 0.001],
                              "gamma": np.logspace(-2, 2, 5)})

    print("Degree %7.1f" % 4)
    print("Tring alphas", [1, 0.01, 0.1, 0.001])
    print("Tring gammas", np.logspace(-2, 2, 5))
    line()
    print("Training...")
    kr.fit(trainx, trainy)
    y_kr = kr.predict(testx)
    line()
    print("Best estimator and R2 score:")
    print("R2 score: %7.5f" %kr.score(testx, testy))
    print(kr.best_estimator_)
    
    limMax = max(float(testy.max()), float(y_kr.max()))
    limMin = min(float(testy.min()), float(y_kr.min()))
    plt.plot(testy, y_kr, '.')
    plt.xlabel('True Values')
    plt.xlim(limMin, limMax)
    plt.ylabel('Predictions')
    plt.ylim(limMin, limMax)
    plt.savefig('KRR'+str(model_seq)+'.png', dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format='png',
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None)
    print("\nKRR y vs y_hat save to :" + 'KRR'+str(model_seq)+'.png')
    
    plt.show()
    
    
concreate_strength = readData()

#controls
if(len(sys.argv) == 1):
    help();
elif int(sys.argv[1]) == 0:
    baseLine(concreate_strength)
elif int(sys.argv[1]) == 1:
    if(len(sys.argv) == 2):
        help()
    try:
        model_seq = int(sys.argv[2])
        mfnn(concreate_strength, model_seq)
    except:
        help();
        sys.exit(-1);
elif int(sys.argv[1]) == 2:
        krr(concreate_strength, 0)

