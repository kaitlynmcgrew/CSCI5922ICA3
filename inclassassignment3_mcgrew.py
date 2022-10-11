#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 15:04:32 2022

@author: kaitlynmcgrew
"""

####### PACKAGES ##############################################################
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

####### LOAD DATA #############################################################
#
data1 = pd.read_csv('pepper_spice_level3D.csv')
data2 = pd.read_csv('tire_score_level2D.csv')


####### MODEL 1: THREE OUTPUTS ################################################
# Model 1: unimportant predictor
# Model is trying to predict how spicy a pepper is (3 levels)
# But one variable (customer review) is not predictive of the label
# It is just a random number between 0 and 100
###############################################################################
# Testing to see how a model performs when information provided is not
# the best/cleanest
print('........... BEGINNING MODEL 1: THREE OUTPUTS ...........  \n')
print('Model 1: unimportant predictor \n')
print('The model is trying to predict how spicy a pepper is (3 levels) \n')
print('but one variable (customer review) is not predictive of the label \n')
print('it is just a random number between 0 and 100 \n')
print('.\n.\n.\n.\n.\n.\n.\n.\n.\n.\n.\n.\n.\n.\n.\n.\n')


data1_shuf = data1.sample(frac=1).reset_index(drop=True)

#Split into X & Y
x_raw = data1_shuf.iloc[:, 1:]
print('This is shape of x \n', x_raw.shape)
y_raw = data1_shuf.iloc[:, 0]
print('This is shape of y \n', y_raw.shape)

#Normalize Inputs using mean normalization
x = (x_raw - x_raw.mean())/x_raw.std()

#One Hot Encode Output
ohe = preprocessing.LabelBinarizer()

#[['Not Spicy', 0], ['Spicy', 1], ['Super Spicy', 2]]
y_labs = [0,1,2]
ohe.fit(y_labs)
y = ohe.transform(y_raw)

#Split into training and testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, 
                                                    random_state=123)

# Create model
model1 = tf.keras.Sequential([
    tf.keras.layers.Dense(4, input_shape=(3,), activation = 'sigmoid'),
    tf.keras.layers.Dense(8, activation = 'softsign'),
    tf.keras.layers.Dense(12, activation = 'softsign'),
    tf.keras.layers.Dense(8, activation = 'softsign'),
    tf.keras.layers.Dense(4, activation = 'softsign'),
    tf.keras.layers.Dense(3, activation = 'softmax')
    ])

model1.compile(optimizer = 'adam',
               loss = 'categorical_crossentropy',
               metrics = ['accuracy'])

# Fit Model
model1.fit(x_train, y_train, epochs = 200)
print('Testing the Model on unseen data')
test_loss, test_acc = model1.evaluate(x_test,  y_test)
print('This is the test loss \n', test_loss)
print('This is the test accuracy \n', test_acc)


####### MODEL 2: TWO OUTPUTS ##################################################
# Model 2: Activation functions
# Observing different activation functions and their outcomes in a set amount
# of epochs
###############################################################################
print('........... BEGINNING MODEL 2: TWO OUTPUTS ...........  \n')
print('Model 2: Activation functions \n')
print('The model is trying to predict if a tire is good (0) or bad (1) \n')
print('Observing different activation functions and their outcomes in a set amount \n')
print('of epochs, hidden layers & nodes \n')
print('.\n.\n.\n.\n.\n.\n.\n.\n.\n.\n.\n.\n.\n.\n.\n.\n')


# Shuffle Data
data2_shuf = data2.sample(frac=1).reset_index(drop=True)

# Split into X & Y
x_raw = data2_shuf.iloc[:, 1:]
print('This is shape of x \n', x_raw.shape)
y = data2_shuf.iloc[:, 0]
print('This is shape of y \n', y.shape)

# Normalize Inputs using mean normalization
x = (x_raw - x_raw.mean())/x_raw.std()

# Split into training and testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, 
                                                    random_state=123)

####### Sigmoid Activation ####################################################
print('Beginning Sigmoid Activation Model')
print('.\n.\n.\n.\n.\n.\n.\n.\n.\n.\n.\n.\n.\n.\n.\n.\n')


model2_sig = tf.keras.Sequential([tf.keras.layers.Dense(10, input_shape=(4,), 
                                   activation = 'sigmoid'),
             tf.keras.layers.Dense(4, activation = 'sigmoid'),
             tf.keras.layers.Dense(1, activation = 'sigmoid')
    ])

model2_sig.compile(optimizer = 'adam',
               loss = keras.losses.MeanSquaredError(),
               metrics = ['accuracy'])

model2_sig.fit(x_train, y_train, epochs = 50)
test_loss_1, test_acc_1 = model2_sig.evaluate(x_test,  y_test)

####### Soft Plus Activation ##################################################
#https://www.tensorflow.org/api_docs/python/tf/keras/activations/softplus
print('Beginning Soft Plus Activation Model')
print('.\n.\n.\n.\n.\n.\n.\n.\n.\n.\n.\n.\n.\n.\n.\n.\n')

model2_softP = tf.keras.Sequential([tf.keras.layers.Dense(10, input_shape=(4,), 
                                     activation = 'softplus'),
             tf.keras.layers.Dense(4, activation = 'softplus'),
             tf.keras.layers.Dense(1, activation = 'softplus')
    ])

model2_softP.compile(optimizer = 'adam',
               loss = keras.losses.MeanSquaredError(),
               metrics = ['accuracy'])

model2_softP.fit(x_train, y_train, epochs = 50)
test_loss_2, test_acc_2 = model2_softP.evaluate(x_test,  y_test)

####### Swish Activation ######################################################
#https://www.tensorflow.org/api_docs/python/tf/keras/activations/swish
print('Beginning Swish Activation Model')
print('.\n.\n.\n.\n.\n.\n.\n.\n.\n.\n.\n.\n.\n.\n.\n.\n')

model2_swish = tf.keras.Sequential([tf.keras.layers.Dense(10, input_shape=(4,), 
                                     activation = 'swish'),
             tf.keras.layers.Dense(4, activation = 'swish'),
             tf.keras.layers.Dense(1, activation = 'swish')
    ])

model2_swish.compile(optimizer = 'adam',
               loss = keras.losses.MeanSquaredError(),
               metrics = ['accuracy'])

model2_swish.fit(x_train, y_train, epochs = 50)
test_loss_3, test_acc_3 = model2_swish.evaluate(x_test,  y_test)

####### Model Evaluation ######################################################

vals = [['sigmoid', test_loss_1, test_acc_1],
        ['soft plus', test_loss_2, test_acc_2],
        ['swish', test_loss_3, test_acc_3]]

results = pd.DataFrame(vals, columns = ['activation', 'loss', 'accuracy'])
print('Here are the results \n', results)

