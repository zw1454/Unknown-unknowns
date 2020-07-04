#!/usr/bin/env python
# coding: utf-8

# In[ ]:


######################################################################################
## Import the relevant libraries
## ==============================
import numpy
import time
import calendar
import pickle
#from six.moves import cPickle as pickle
import numpy as np
import cv2
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import random
import numpy as np
import tensorflow as tf
#import keras
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Activation, BatchNormalization
#from tensorflow.keras.utils import np_utils
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.backend import sigmoid
from tensorflow.keras.utils import get_custom_objects
import efficientnet.tfkeras as enet
from tensorflow.keras import regularizers, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import warnings
import random
import numpy as np
#import matplotlib.pyplot as plt
from numpy.random import multivariate_normal
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
import math

def make_data(uu_label, uu_label_test):
    (old_x_train, old_y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    
    # Making sure that the values are float so that we can get decimal points after division
    old_x_train = old_x_train.astype('float32')
    x_test = x_test.astype('float32')
    
    # Normalizing the RGB codes by dividing it to the max RGB value.
    old_x_train /= 255
    x_test /= 255
    
    
    print('x_train shape:', old_x_train.shape)
    print('Number of images in x_train: ', old_x_train.shape[0])
    print('Number of images in x_test: ', x_test.shape[0])
        
    for i in range(y_test.shape[0]):
        if y_test[i] in uu_label:
            y_test[i] = uu_label_test
    
    x_train = []
    y_train = []
    uu_x = []
    uu_y = []
    for i in range(old_x_train.shape[0]):
        if old_y_train[i] not in uu_label:
            x_train.append(old_x_train[i])
            y_train.append(old_y_train[i])
        else:
            uu_x.append(old_x_train[i])
            uu_y.append(old_y_train[i])
    

                   
    x_train = np.array(x_train)     
    y_train = np.array(y_train)
#     print(np.unique(y_train))
    uu_x = np.array(uu_x)
    uu_y = np.array(uu_y)
    

    print("Shape of all data: ", x_train.shape[0]+uu_x.shape[0])
    print("Shape of unknown unknown data: ", uu_x.shape[0])
    
    zipped = list(zip(x_test, y_test))  
    random.shuffle(zipped)
    x_test, y_test = zip(*zipped)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

# In case need to test running model on sets of known 
# and unknown classes that do not contain consecutively
#numbered classes, we need to relabel both known and unknown 
#sets such that they contain consecutive classes since the DNN 
#architecture is not a Sequential one. 
#Example for known = [0,1,2,3,8,9] and U.U = [4,5,6,7] :
       
#    for i in range(x_train.shape[0]):
 #       if y_train[i] == 8:
#            y_train[i] = 4
 #       elif y_train[i] == 9:
#            y_train[i] = 5

            
 #   for i in range(x_test.shape[0]):
  #      if y_test[i] == 8:
   #         y_test[i] = 4
    #    elif y_test[i] == 9:
     #       y_test[i] = 5
   
           

    
    
    return x_train, x_test, y_train, y_test

'''------------------- phase 1 -------------------------
In this phase, we will train the neural network on the known classes
and test its performance on the test classes.
'''

def CNN_phase1(x_train, x_test, y_train, y_test, uu_label, uu_label_test):
    # Creating network architecture
    


    baseMapNum = x_train.shape[1]
    weight_decay = 1e-4
    model = Sequential()
    model.add(Conv2D(baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(2*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(2*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(4*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(4*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(uu_label_test+1, activation='softmax'))

    #model.summary()

#data augmentation
    datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=False
    )
    datagen.fit(x_train)

#training
    batch_size = 64
    epochs=7
    opt_rms = optimizers.RMSprop(lr=0.001,decay=1e-6)
    model.compile(loss='sparse_categorical_crossentropy',
        optimizer=opt_rms,
        metrics=['accuracy'])
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),steps_per_epoch=x_train.shape[0] // batch_size,epochs=3*epochs,verbose=1,validation_data=(x_test,y_test))
    model.save_weights('cifar10_normal_rms_ep21-1.h5')

    opt_rms = optimizers.RMSprop(lr=0.0005,decay=1e-6)
    model.compile(loss='sparse_categorical_crossentropy',
        optimizer=opt_rms,
        metrics=['accuracy'])
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),steps_per_epoch=x_train.shape[0] // batch_size,epochs=epochs,verbose=1,validation_data=(x_test,y_test))
    model.save_weights('cifar10_normal_rms_ep28-1.h5')

    opt_rms = optimizers.RMSprop(lr=0.0003,decay=1e-6)
    model.compile(loss='sparse_categorical_crossentropy',
        optimizer=opt_rms,
        metrics=['accuracy'])
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),steps_per_epoch=x_train.shape[0] // batch_size,epochs=epochs,verbose=1,validation_data=(x_test,y_test))
    model.save_weights('cifar10_normal_rms_ep35-1.h5')

    
    
    score1 = model.evaluate(x_test, y_test)[1]
    print("\nOriginal accuracy score: ", score1)
    
    origin_pred = model.predict_classes(x_test)
    f_measure = f1_score(y_test, origin_pred, average='macro')
    print("\nOriginal F-measure: ", f_measure)
    
    return (score1, f_measure)
    
'''------------------- phase 2 ------------------------
In this phase, we will: first, random sample from the test set and add it as a new
class to the existing training set; second, perform a cross validation on the training
set and sort out all the data assigned to the new sample class; third, retrain the classifer
on the new training set and evaluate its performance on the test set.
'''

def random_sampling(n, x_train, x_test, y_train, y_test):      
    # n: number of samples from the test set
    A = [list(i) for i in x_test]
    a = list(y_test) 
    newA, newa = zip(*random.sample(list(zip(A, a)), n))
    newA_l, newa_l = list(newA), list(newa)
    sample, sample_label = np.array(newA_l), np.array(newa_l) # obtain a sample of test set
    


    print("\nRandom sampling complete!")
    return sample, sample_label, x_test, y_test

def cross_validation(X, Y, uu_label, uu_label_test):
    # divide the training data into k-folds
    kf = KFold(n_splits=3)
    unknown = []
    unknown_label = []
    uu_count = 0
    
    for train_index, test_index in kf.split(X, Y):
        xc_train, xc_test = X[train_index], X[test_index]
        yc_train, yc_test = Y[train_index], Y[test_index]
        
        baseMapNum = xc_train.shape[1]
        weight_decay = 1e-4
        model = Sequential()
        model.add(Conv2D(baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=xc_train.shape[1:]))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(2*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(2*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.3))

        model.add(Conv2D(4*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(4*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.4))

        model.add(Flatten())
        model.add(Dense(uu_label_test+1, activation='softmax'))

#    model.summary()

#data augmentation
        datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=False
       )
        datagen.fit(xc_train)

#training
        batch_size = 64
        epochs=7
        opt_rms = optimizers.RMSprop(lr=0.001,decay=1e-6)
        model.compile(loss='sparse_categorical_crossentropy',
        optimizer=opt_rms,
        metrics=['accuracy'])
        model.fit_generator(datagen.flow(xc_train, yc_train, batch_size=batch_size),steps_per_epoch=xc_train.shape[0] // batch_size,epochs=3*epochs,verbose=1,validation_data=(xc_test,yc_test))
        model.save_weights('cifar10_normal_rms_ep21-c.h5')

        opt_rms = optimizers.RMSprop(lr=0.0005,decay=1e-6)
        model.compile(loss='sparse_categorical_crossentropy',
        optimizer=opt_rms,
        metrics=['accuracy'])
        model.fit_generator(datagen.flow(xc_train, yc_train, batch_size=batch_size),steps_per_epoch=xc_train.shape[0] // batch_size,epochs=epochs,verbose=1,validation_data=(xc_test,yc_test))
        model.save_weights('cifar10_normal_rms_ep28-c.h5')

        opt_rms = optimizers.RMSprop(lr=0.0003,decay=1e-6)
        model.compile(loss='sparse_categorical_crossentropy',
        optimizer=opt_rms,
        metrics=['accuracy'])
        model.fit_generator(datagen.flow(xc_train, yc_train, batch_size=batch_size),steps_per_epoch=xc_train.shape[0] // batch_size,epochs=epochs,verbose=1,validation_data=(xc_test,yc_test))
        model.save_weights('cifar10_normal_rms_ep35-c.h5')
    
    
        # sort out the samples classified into the new class
        for x in xc_test:
            l = model.predict_classes(np.array([x]))
            if l == np.array([uu_label_test]):                                    ####
                uu_count += 1
                unknown.append(list(x))
                unknown_label.append(uu_label_test)                               #####
    
#    print("\nThe number of elements in the unknown unknown class: ", uu_count)
    unknown = np.array(unknown)
    unknown_label = np.array(unknown_label)
#    print(unknown.shape)
    
    return unknown, unknown_label

    
def CNN_phase2(sample_size, x_train, x_test, y_train, y_test, uu_label, uu_label_test):
    sample_num = int(sample_size * y_test.shape[0])
    
    sample, sample_label, new_x_test, new_y_test             = random_sampling(sample_num, x_train, x_test, y_train, y_test)
            
    sample_label = [uu_label_test for i in range(len(sample_label))]              #####
    sample_label = np.array(sample_label)
    
    classes = np.concatenate((x_train, sample), axis=0) # construct new training classes
    classes_label = np.append(y_train, sample_label)
    
    # randomly shuffle the classes
    zipped = list(zip(classes, classes_label))  
    random.shuffle(zipped)
    classes, classes_label = zip(*zipped)
    classes = np.array(classes)
    classes_label = np.array(classes_label)
    
    un_class, un_label = cross_validation(classes, classes_label, uu_label, uu_label_test)
    
    print(un_class.shape)
    if un_class.shape[0] != 0:
         x_train2 = np.concatenate((x_train, un_class), axis=0)
    else:
        x_train2 = x_train
        
   
    y_train2 = np.append(y_train, un_label)    
    
    baseMapNum = x_train2.shape[1]
    weight_decay = 1e-4
    model = Sequential()
    model.add(Conv2D(baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=x_train2.shape[1:]))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(2*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(2*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(4*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(4*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(uu_label_test+1, activation='softmax'))

    #model.summary()

#data augmentation
    datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=False
    )
    datagen.fit(x_train2)

#training
    batch_size = 64
    epochs=7
    opt_rms = optimizers.RMSprop(lr=0.001,decay=1e-6)
    model.compile(loss='sparse_categorical_crossentropy',
        optimizer=opt_rms,
        metrics=['accuracy'])
    model.fit_generator(datagen.flow(x_train2, y_train2, batch_size=batch_size),steps_per_epoch=x_train2.shape[0] // batch_size,epochs=3*epochs,verbose=1,validation_data=(new_x_test,new_y_test))
    model.save_weights('cifar10_normal_rms_ep21-2.h5')

    opt_rms = optimizers.RMSprop(lr=0.0005,decay=1e-6)
    model.compile(loss='sparse_categorical_crossentropy',
        optimizer=opt_rms,
        metrics=['accuracy'])
    model.fit_generator(datagen.flow(x_train2, y_train2, batch_size=batch_size),steps_per_epoch=x_train2.shape[0] // batch_size,epochs=epochs,verbose=1,validation_data=(new_x_test,new_y_test))
    model.save_weights('cifar10_normal_rms_ep28-2.h5')

    opt_rms = optimizers.RMSprop(lr=0.0003,decay=1e-6)
    model.compile(loss='sparse_categorical_crossentropy',
        optimizer=opt_rms,
        metrics=['accuracy'])
    model.fit_generator(datagen.flow(x_train2, y_train2, batch_size=batch_size),steps_per_epoch=x_train2.shape[0] // batch_size,epochs=epochs,verbose=1,validation_data=(new_x_test,new_y_test))
    model.save_weights('cifar10_normal_rms_ep35-2.h5')
    
    
    
    
    
    score2 = model.evaluate(new_x_test, new_y_test)[1]
    print("\nFinal accuracy score: ", score2)
    
    origin_pred = model.predict_classes(new_x_test)
    print(origin_pred)
    print(origin_pred.shape)
    f_measure = f1_score(new_y_test, origin_pred, average='macro')
    print("\nFinal F-measure: ", f_measure)
    
    
    return score2, f_measure
  
def CNN_main(sample_size, uu_label, uu_label_test):
    x_train, x_test, y_train, y_test = make_data(uu_label, uu_label_test)
    score1, f_measure1 = CNN_phase1(x_train, x_test, y_train, y_test, uu_label, uu_label_test)
    score2, f_measure2 = CNN_phase2(sample_size, x_train, x_test, y_train, y_test, uu_label, uu_label_test)
    return score1, f_measure1, score2, f_measure2


score1, f_measure1, score2, f_measure2 = CNN_main(0.3, [9], 9) 

with open('uuu.pickle', 'wb') as f: #storing results
        pickle.dump([score1 , f_measure1, score2, f_measure2], f)
    
    


