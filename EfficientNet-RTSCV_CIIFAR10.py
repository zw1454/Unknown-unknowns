#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import random
import numpy as np
import tensorflow as tf
import keras
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Activation, BatchNormalization
from keras.utils import np_utils
from keras.datasets import cifar10
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.backend import sigmoid
from keras.utils.generic_utils import get_custom_objects
import efficientnet.keras as enet
import warnings
warnings.filterwarnings("ignore")

#pre-reqs for building EfficientNet neural network architecture
get_ipython().system('pip install -U git+https://github.com/qubvel/efficientnet')
#custom activation function used in the DNN    
class SwishActivation(Activation):
    
    def __init__(self, activation, **kwargs):
        super(SwishActivation, self).__init__(activation, **kwargs)
        self.__name__ = 'swish_act'

def swish_act(x, beta = 1):
    return (x * sigmoid(beta * x))

get_custom_objects().update({'swish_act': SwishActivation(swish_act)})

# Global variables to be used in the DNN
batch_size = 32

input_shape = (32, 32, 3)


'''------------------ make data ------------------------ 
In this part, we generate our data and perform
a multidimensional scaling (MDS) to visualize the class separability.
Then we split our generated data into test/training sets.
'''

def make_data(uu_label, uu_label_test):
    (old_x_train, old_y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

#     Reshaping the array to 4-dims so that it can work with the Keras API
#     old_x_train = old_x_train.reshape(old_x_train.shape[0], 32, 32, 1)
#     x_test = x_test.reshape(x_test.shape[0], 32, 32, 1)
    
    # Making sure that the values are float so that we can get decimal points after division
    old_x_train = old_x_train.astype('float32')
    x_test = x_test.astype('float32')
    
    # Normalizing the RGB codes by dividing it to the max RGB value.
    old_x_train /= 255
    x_test /= 255
    
#     old_y_train = np_utils.to_categorical(old_y_train, )
#     y_test = np_utils.to_categorical(y_test, 10)
#     print(y_test)

    
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

#     for i in range(x_train.shape[0]):
#         if y_train[i] == 8:
#             y_train[i] = 4
#         elif y_train[i] == 9:
#             y_train[i] = 5
#     for i in range(x_test.shape[0]):
#         if y_test[i] == 8:
#             y_test[i] = 4
#         elif y_test[i] == 9:
#             y_test[i] = 5
    
    
    return x_train, x_test, y_train, y_test

'''------------------- phase 1 -------------------------
In this phase, we will train the neural network on the known classes
and test its performance on the test classes.
'''

def CNN_phase1(x_train, x_test, y_train, y_test, uu_label, uu_label_test):
    # Creating an EfficientNetB0 base Model and adding 2 fully connected layers for fine tuning

    model = enet.EfficientNetB0(include_top=False, input_shape=(32,32,3), pooling='avg', weights='imagenet')

# Adding 2 fully-connected layers to B0.
    x = model.output

    x = BatchNormalization()(x)
    x = Dropout(0.7)(x)

    x = Dense(512)(x)
    x = BatchNormalization()(x)
    x = Activation(swish_act)(x)
    x = Dropout(0.5)(x)

    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = Activation(swish_act)(x)

# Output layer
    predictions = Dense(uu_label_test+1, activation="softmax")(x)

    model_final = Model(inputs = model.input, outputs = predictions)

    #model_final.summary()
    
    model_final.compile(optimizer=Adam(0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    mcp_save = ModelCheckpoint(r'C:\Users\IST\Downloads\EnetB0_CIFAR10p1_TL.h5', save_best_only=True, monitor='val_acc') #saving model
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=2, verbose=1,) # reduce learning rate during training if needed

    model_final.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=10,
              validation_data=(x_test, y_test),
              shuffle=True,
              callbacks=[mcp_save, reduce_lr])
    
    score1 = model_final.evaluate(x_test, y_test)[1]
    print("\nOriginal accuracy score: ", score1)
    
    origin_pred = np.argmax(model_final.predict(x_test), axis=1)
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
        

        
        model = enet.EfficientNetB0(include_top=False, input_shape=(32,32,3), pooling='avg', weights='imagenet')

# Adding 2 fully-connected layers to B0.
        x = model.output

        x = BatchNormalization()(x)
        x = Dropout(0.7)(x)

        x = Dense(512)(x)
        x = BatchNormalization()(x)
        x = Activation(swish_act)(x)
        x = Dropout(0.5)(x)

        x = Dense(128)(x)
        x = BatchNormalization()(x)
        x = Activation(swish_act)(x)

# Output layer
        predictions = Dense(uu_label_test+1, activation="softmax")(x)

        model_final = Model(inputs = model.input, outputs = predictions)

    #model_final.summary()
    
        model_final.compile(optimizer=Adam(0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        mcp_save = ModelCheckpoint(r'C:\Users\IST\Downloads\EnetB0_CIFAR10cv_TL.h5', save_best_only=True, monitor='val_acc')
        reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=2, verbose=1,)

        model_final.fit(x=xc_train, y=yc_train,
              batch_size=batch_size,
              epochs=10,
              validation_data=(xc_test, yc_test),
              shuffle=True,
              callbacks=[mcp_save, reduce_lr])
    
    
        # sort out the samples classified into the new class
        for x in xc_test:
            l = np.argmax(model_final.predict(np.array([x])), axis=1)
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
    
    
    # construct the final training set by adding the unknown unknowns
    print(x_train2.shape)
    print(un_class.shape)
    x_train2 = np.concatenate((x_train, un_class), axis=0)
    y_train2 = np.append(y_train, un_label)    
    
    # train & test on the new sets

    
    model = enet.EfficientNetB0(include_top=False, input_shape=(32,32,3), pooling='avg', weights='imagenet')

# Adding 2 fully-connected layers to B0.
    x = model.output

    x = BatchNormalization()(x)
    x = Dropout(0.7)(x)

    x = Dense(512)(x)
    x = BatchNormalization()(x)
    x = Activation(swish_act)(x)
    x = Dropout(0.5)(x)

    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = Activation(swish_act)(x)

# Output layer
    predictions = Dense(uu_label_test+1, activation="softmax")(x)

    model_final = Model(inputs = model.input, outputs = predictions)

    #model_final.summary()
    
    model_final.compile(optimizer=Adam(0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    mcp_save = ModelCheckpoint(r'C:\Users\IST\Downloads\EnetB0_CIFAR10p2_TL.h5', save_best_only=True, monitor='val_acc')
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=2, verbose=1,)

    model_final.fit(x=x_train2, y=y_train2,
              batch_size=batch_size,
              epochs=9,
              validation_data=(new_x_test, new_y_test),
              shuffle=True,
              callbacks=[mcp_save, reduce_lr])
    
    
    
    score2 = model_final.evaluate(new_x_test, new_y_test)[1]
    print("\nFinal accuracy score: ", score2)
    
    origin_pred = np.argmax(model_final.predict(new_x_test), axis=1)
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

 
# ####################################
score1, f_measure1, score2, f_measure2 = CNN_main(0.3, [6,7,8,9], 6) 
score3, f_measure3, score4, f_measure4 = CNN_main(0.3, [4,5,6,7,8,9], 4) 
score5, f_measure5, score6, f_measure6 = CNN_main(0.3, [2,3,4,5,6,7,8,9], 2) 
with open('uu.pickle', 'wb') as f: #storing results
             pickle.dump([score1, f_measure1, score2, f_measure2, score3, f_measure3, score4, f_measure4, score5, f_measure5, score6, f_measure6], f)


# In[ ]:




