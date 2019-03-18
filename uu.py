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


'''------------------ make data ------------------------ 
In this part, we generate our data (10 classes + 1 unknown unknown class) and perform
a multidimensional scaling (MDS) to visualize the class separability.
Then we split our generated data into test/training sets.
'''

x, y = make_blobs(n_samples=1000, n_features=20, centers=10,\
            cluster_std=1.5, center_box=(-7.0, 7.0), shuffle=True, \
            random_state=20)          # the label of y ranging from 0 to 9

ux, uy = make_blobs(n_samples=100, n_features=20, centers=1,\
            cluster_std=3, center_box=(-10, 10), shuffle=True, \
            random_state=20)
uy = np.array([10 for i in range(len(uy))])  # set the label of unknown unknowns as 10

all_x, all_y = np.concatenate((x, ux), axis=0), np.concatenate((y, uy), axis=0)


print("Performing Multidimensional Scaling...")
embedding = MDS(n_components=2)
all_x_transformed = embedding.fit_transform(all_x)

pyplot.figure(1)
pyplot.scatter(all_x_transformed[:,0],all_x_transformed[:,1],c=all_y)
pyplot.title("The structure of the 10+uu classes")
pyplot.show()  # Note that the color of the unknown unknown class is yellow

x = preprocessing.normalize(x, norm='l2')
ux = preprocessing.normalize(ux, norm='l2')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
x_test = np.concatenate((x_test, ux))
y_test = np.concatenate((y_test, uy))



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
    print("\nOriginal test score: \n", model.score(x_test, y_test))
    
    
    
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
    print("Shape of sample class: ", sample.shape)
    print("Shape of sample label: ", sample_label.shape)
    print("Shape of new test set: ", new_x_test.shape)
    print("Shape of new test label: ", new_y_test.shape)
    
    return sample, sample_label, new_x_test, new_y_test

def cross_validation(X, Y, k=10):
    # divide the training data into k-folds
    group = np.reshape(X, (k,-1))      # group.shape = (10, 20*76)
    group_labels = np.reshape(Y, (k,-1))     # group_lable.shape = (10, 76)
    

    
def phase2():
    sample, sample_label, new_x_test, new_y_test = random_sampling(60)
    sample_label = [10 for i in range(len(sample_label))]
    sample_label = np.array(sample_label)
    
    classes = np.concatenate((x_train, sample), axis=0) # construct new training classes
    classes_label = np.append(y_train, sample_label)
    print("Shape of new training classes: ", classes.shape)
    print("Shape of new training label: ", classes_label.shape)
    
    zipped = list(zip(classes, classes_label))  # randomly shuffle the classes
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
    
    cross_validation(classes, classes_label, k=10)
 
    # use probabilistic classifier
    # confusion matrix
  
    





SVM_phase1()
phase2()
    



#    def phase2():
#    sample, sample_label, new_x_test, new_y_test = random_sampling(60)
#    A = [list(i) for i in x_test]
#    a = list(y_test) 
#    dic = {}
#    for i in range(len(x_test_l)):
#        dic[str(x_test_l[i])] = y_test_l[i]
#        
#    sample_index = random.sample([i for i in range(len(x_test_l))], 100)
#    sample = []
#    for i in sample_index:
#        sample.append(x_test_l[i])
#    
#    for s in sample:
#        x_test_l.remove(s)
#    sample_index.sort()
#    k = 0
#    for i in sample_index:
#        del y_test_l[i-k]
#        k += 1
#        
#    x_test1 = np.array(x_test_l)
#    y_test1 = np.array(y_test_l)
    
#    sample_label = [10]*100
#    
#    sample = np.array(sample)
#    sample_lable = np.array(sample_label)
#    
#    classes = np.concatenate((x_train, sample), axis=0)
#    classes_label = np.append(y_train, sample_label)
#    
#    # start cross validation
#    model = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,             # confusion matrix
#                decision_function_shape='ovr', degree=3, gamma='auto',           # use probabilistic classifier
#                kernel='rbf', max_iter=-1, probability=False, random_state=None,
#                shrinking=True, tol=0.001, verbose=False)
#    model.fit(classes, classes_label)
#    scores = cross_val_score(model, classes, classes_label, cv=11)
#    print("Cross validation scores: ", scores)


    