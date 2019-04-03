import os
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import StratifiedKFold
import preprocessor as p
from nltk import PorterStemmer 
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import string
import pandas as pd
from stw import SupervisedTermWeightingWTransformer
from numpy import array

import sys
import ast
from collections import Counter
from os import listdir
import simplejson
import math
import numpy as np

###To find out frequency of documents that contain a particular term in the vocabulary###
def document_frequency(pos_documents,neg_documents,vocabulary_list):
	pos_documents_freq=[]
	neg_documents_freq=[]
	for word in vocabulary_list:
		pos_count=0
		neg_count=0
		for document in pos_documents:
			if word in document:
				pos_count+=1
		pos_documents_freq.append(pos_count)

		for document in neg_documents:
			if word in document:
				neg_count+=1
		neg_documents_freq.append(neg_count)

	return pos_documents_freq,neg_documents_freq

###MI for Positive Corpus###
def MI_for_positive_corpus(pos_documents,neg_documents,pos_documents_freq,neg_documents_freq):
	pos_D=len(pos_documents)
	neg_D=len(neg_documents)
	D=pos_D+neg_D
	MI_pos=[]
	for i in range(len(pos_documents_freq)):
		numerator=pos_documents_freq[i] * D
		denominator=(pos_documents_freq[i]+neg_documents_freq[i])*len(pos_documents)

		if denominator == 0 or float(numerator)/denominator == 0:
			MI_pos.append(0)
		else:
			MI_per_term = float(numerator)/denominator
			MI_per_term=math.log(MI_per_term,2)
			MI_pos.append(MI_per_term)

	return MI_pos

###MI for Negative Corpus###
def MI_for_negative_corpus(pos_documents,neg_documents,pos_documents_freq,neg_documents_freq):
	pos_D=len(pos_documents)
	neg_D=len(neg_documents)
	D=pos_D+neg_D
	MI_neg=[]
	for i in range(len(neg_documents_freq)):
		numerator=neg_documents_freq[i] * D
		denominator=(pos_documents_freq[i]+neg_documents_freq[i])*len(neg_documents)

		if denominator == 0 or float(numerator)/denominator == 0:
			MI_neg.append(0)
		else:
			MI_per_term = float(numerator)/denominator
			MI_per_term=math.log(MI_per_term,2)
			MI_neg.append(MI_per_term)

	return MI_neg

###Calculating MI###
def MI(MI_pos,MI_neg):
	MI_result=[]
	for i in range(len(MI_pos)):
		MI_result.append(max(MI_pos[i],MI_neg[i]))

	return MI_result

def MI_mapper(MI_result,vocabulary_list):
    d={}
    for i in range(len(vocabulary_list)):
        d[vocabulary_list[i]]=MI_result[i]
    return d

file_path='pos_corpus.txt'
f1 = open(file_path, 'r')
pos_corpus=f1.read()
pos_corpus = ast.literal_eval(pos_corpus)
f1.close()

file_path='neg_corpus.txt'
f1 = open(file_path, 'r')
neg_corpus=f1.read()
neg_corpus = ast.literal_eval(neg_corpus)
f1.close()

file_path='corpus.txt'
f1 = open(file_path, 'r')
corpus=f1.read()
corpus = ast.literal_eval(corpus)
f1.close()

file_path='food_vocabulary_list.txt'
f1 = open(file_path, 'r')
food_vocabulary_list=f1.read()
food_vocabulary_list = ast.literal_eval(food_vocabulary_list)
f1.close()

pos_documents_freq,neg_documents_freq=document_frequency(pos_corpus,neg_corpus,food_vocabulary_list)
MI_pos=MI_for_positive_corpus(pos_corpus,neg_corpus,pos_documents_freq,neg_documents_freq)
MI_neg=MI_for_negative_corpus(pos_corpus,neg_corpus,pos_documents_freq,neg_documents_freq)
MI_result=MI(MI_pos,MI_neg)
d = MI_mapper(MI_result,food_vocabulary_list)

labels = np.zeros(249292);
labels[0:124646]=1;
labels[124646:]=0; 
       
kf = StratifiedKFold(n_splits=10)
 
totalsvm = 0           # Accuracy measure on 2000 files
totalNB = 0
totalLR = 0
totalMatSvm = np.zeros((2,2));  # Confusion matrix on 2000 files
totalMatNB = np.zeros((2,2));
totalMatLR = np.zeros((2,2));

for train_index, test_index in kf.split(corpus,labels):
    X_train = [corpus[i] for i in train_index]
    X_test = [corpus[i] for i in test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    
    MI_train=[]
    for i in range(len(X_train)):
        score=[]
        for j in range(len(food_vocabulary_list)):
            if food_vocabulary_list[j] in X_train[i]:
                score.append(d[food_vocabulary_list[j]])
            else:
                score.append(0.0)
        MI_train.append(score)
    
    print("MI Train Done")
        
    
    MI_test=[]
    for i in range(len(X_test)):
        score=[]
        for j in range(len(food_vocabulary_list)):
            if food_vocabulary_list[j] in X_test[i]:
                score.append(d[food_vocabulary_list[j]])
            else:
                score.append(0.0)
        MI_test.append(score)
    
    print("MI Test Done")

    
    model1 = LinearSVC()
    model2 = MultinomialNB()   
    model3 = LogisticRegression()
    model1.fit(MI_train,y_train)
    model2.fit(MI_train,y_train)
    model3.fit(MI_train,y_train)
    result1 = model1.predict(MI_test)
    result2 = model2.predict(MI_test)
    result3 = model3.predict(MI_test)
    
     
    totalMatSvm = totalMatSvm + confusion_matrix(y_test, result1)
    totalMatNB = totalMatNB + confusion_matrix(y_test, result2)
    totalMatLR = totalMatLR + confusion_matrix(y_test, result3)
    totalsvm = totalsvm+sum(y_test==result1)
    totalNB = totalNB+sum(y_test==result2)
    totalLR = totalLR+sum(y_test==result3)

print("########Results########")
print("SVM: ",totalMatSvm, totalsvm/249292.0)
print("NB: ",totalMatNB, totalNB/249292.0)
print("LR: ",totalMatLR, totalLR/249292.0)

print()
print()
from sklearn.metrics import f1_score
print("SVM",f1_score(y_test, result1, average='binary')) 
print("NB",f1_score(y_test, result2, average='binary')) 
print("LR",f1_score(y_test, result3, average='binary')) 