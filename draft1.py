#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 23:41:28 2019

@author: Wang, Zheng (zw1454@nyu.edu)
"""
import random
import numpy as np
from sklearn.datasets import make_blobs
from matplotlib import pyplot
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.manifold import MDS
from sklearn.model_selection import KFold

'''------------------ make data ------------------------ 
In this part, we generate our data (10 classes + 1 unknown unknown class) and perform
a multidimensional scaling (MDS) to visualize the class separability.
Then we split our generated data into test/training sets.
'''

x, y = make_blobs(n_samples=1000, n_features=20, centers=10,\
            cluster_std=1.5, center_box=(-7.0, 7.0), shuffle=True, \
            random_state=20)          # the label of y ranging from 0 to 9

ux, uy = make_blobs(n_samples=200, n_features=20, centers=1,\
            cluster_std=2, center_box=(-7, 2), shuffle=True, \
            random_state=20)
uy = np.array([10 for i in range(len(uy))])  # set the label of unknown unknowns as 10

all_x, all_y = np.concatenate((x, ux), axis=0), np.concatenate((y, uy), axis=0)

print("Shape of all data: ", all_x.shape)
print("Shape of unknown unknown data: ", ux.shape)

x = preprocessing.normalize(x, norm='l2')
ux = preprocessing.normalize(ux, norm='l2')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
x_test = np.concatenate((x_test, ux))
y_test = np.concatenate((y_test, uy))

print("Shape of the original training data: ", x_train.shape)
print("Shape of the original test data: ", x_test.shape)

print("Performing Multidimensional Scaling...")
embedding = MDS(n_components=2)
all_x_transformed = embedding.fit_transform(all_x)

pyplot.figure(1)
pyplot.scatter(all_x_transformed[:,0],all_x_transformed[:,1],c=all_y)
pyplot.title("The structure of the 10+uu classes")
pyplot.show()  # Note that the color of the unknown unknown class is yellow

'''------------------- phase 1 -------------------------
In this phase, we will train the classifier (here we pick SVM) on the 10 known classes
and test its performance on the 10+1 test classes.
This phase serves as the control group of our experiment.
'''

def SVM_phase1():    
    model = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto',
      kernel='rbf', max_iter=-1, probability=False, random_state=None,
      shrinking=True, tol=0.001, verbose=False)
    
    model.fit(x_train, y_train)
    model.predict(x_test)
    print("\nOriginal test score: ", model.score(x_test, y_test))
    
    
    
'''------------------- phase 2 ------------------------
In this phase, we will: first, random sample from the test set and add it as a new
class to the existing training set; second, perform a cross validation on the training
set and sort out all the data assigned to the new sample class; third, retrain the classifer
on the new 10+1 class training set and evaluate its performance on the test set.
This phase is the experiment group.
'''

def random_sampling(n):       # n: number of samples from the test set
    A = [list(i) for i in x_test]
    a = list(y_test) 
    newA, newa = zip(*random.sample(list(zip(A, a)), n))
    newA_l, newa_l = list(newA), list(newa)
    sample, sample_label = np.array(newA_l), np.array(newa_l) # obtain a sample of test set
    
    new_x_test = []
    new_y_test = []
    for i in range(len(A)):
        if A[i] not in sample:
            new_x_test.append(A[i])
            new_y_test.append(a[i]) # reconstruct the test set by removing the samples
            
    new_x_test = np.array(new_x_test)
    new_y_test = np.array(new_y_test)
    print("\nShape of sample class: ", sample.shape)
    print("Shape of new test set: ", new_x_test.shape)
    
    return sample, sample_label, new_x_test, new_y_test

def cross_validation(X, Y, k=10):
    # divide the training data into k-folds
    kf = KFold(n_splits=k)
    unknown = []
    unknown_label = []
    uu_count = 0
    
    for train_index, test_index in kf.split(X, Y):
        xc_train, xc_test = X[train_index], X[test_index]
#        print("\nShape of cross validation training class: ", xc_train.shape)
#        print("\nShape of cross validation test class: ", xc_test.shape)
        yc_train, yc_test = Y[train_index], Y[test_index]
        
        cross_model = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
             decision_function_shape='ovr', degree=3, gamma='auto',
             kernel='rbf', max_iter=-1, probability=True, random_state=None,
             shrinking=True, tol=0.001, verbose=False)
        cross_model.fit(xc_train, yc_train)
        
        for x in xc_test:
            a = cross_model.predict(np.array([x]))
            if a == np.array([10]):
                uu_count += 1
                unknown.append(list(x))
                unknown_label.append(10)
                
    print("The number of elements in the unknown unknown class: ", uu_count)
    unknown = np.array(unknown)
    unknown_label = np.array(unknown_label)
    print(unknown.shape)
    
    return unknown, unknown_label

    # use probabilistic classifier
    # confusion matrix

    
def phase2():
    sample, sample_label, new_x_test, new_y_test = random_sampling(100)
    sample_label = [10 for i in range(len(sample_label))]
    sample_label = np.array(sample_label)
    
    classes = np.concatenate((x_train, sample), axis=0) # construct new training classes
    classes_label = np.append(y_train, sample_label)
    print("Shape of new training classes: ", classes.shape)
    
    # randomly shuffle the classes
    zipped = list(zip(classes, classes_label))  
    random.shuffle(zipped)
    classes, classes_label = zip(*zipped)
    classes = np.array(classes)
    classes_label = np.array(classes_label)
    
    embedding = MDS(n_components=2)
    c_transformed = embedding.fit_transform(classes)
    
    pyplot.figure(2)
    pyplot.scatter(c_transformed[:,0],c_transformed[:,1],c=classes_label)
    pyplot.title("The structure of the new training 10+sample classes")
    pyplot.show()  # Note that the color of the sample class is yellow
    
    un_class, un_label = cross_validation(classes, classes_label, k=10)
    
    # construct the final training set by adding the unknown unknowns
    x_train2 = np.concatenate((x_train, un_class), axis=0)
    y_train2 = np.append(y_train, un_label)
    print("\nShape of final training classes: ", x_train2.shape)
    print("Shape of final test data: ", new_x_test.shape)
    
    # train & test on the new sets
    model2 = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto',
      kernel='rbf', max_iter=-1, probability=False, random_state=None,
      shrinking=True, tol=0.001, verbose=False)
    
    model2.fit(x_train2, y_train2)
    model2.predict(new_x_test)
    print("\nFinal test score: ", model2.score(new_x_test, new_y_test))
  
def main():
    SVM_phase1()
    phase2()
            
main()
    
