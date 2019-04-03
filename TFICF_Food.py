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

###Term Frequency###
def term_frequency(documents,vocabulary_list):
	TF=[]
	for document in documents:
		tf_per_document=[]
		for word in vocabulary_list:
			fij=document.count(word)
			if fij>0:
				tf=1+math.log(fij,2)
			else:
				tf=0
			tf_per_document.append(tf)
		TF.append(tf_per_document)

	return TF

###Term Frequency According to paper###
def term_frequency_paper(documents,vocabulary_list):
	TF=[]

	for document in documents:
		N=len(document)
		tf_per_document=[]
		for word in vocabulary_list:
			fij=document.count(word)

			tf=float(fij)/N
			tf_per_document.append(tf)
		TF.append(tf_per_document)

	return TF

###Inverse Class Frequency###
def inverse_class_frequency(pos_documents,neg_documents,vocabulary_list):
	N=2
	ICF=[]
	for word in vocabulary_list:
		count_pos=0
		count_neg=0
		for document in pos_documents:
			if word in document:
				count_pos=1
		for document in neg_documents:
			if word in document:
				count_neg=1
		count=count_pos+count_neg
		if count == 0:
			icf=0
		else:
			icf=math.log(N/count,2)
		ICF.append(icf)
	return ICF


###Term Frequency Inverse Class Frequency###
def term_frequency_inverse_class_frequency(TF,ICF):
	ICF=np.array(ICF)
	TFICF=[]
	for tf in TF:
		tf=np.array(tf)
		tficf=tf*ICF
		TFICF.append(tficf.tolist())
	return TFICF



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
    
    
    TF=term_frequency_paper(X_train,food_vocabulary_list)
    ICF=inverse_class_frequency(pos_corpus,neg_corpus,food_vocabulary_list)
    train_corpus_tf_icf=term_frequency_inverse_class_frequency(TF,ICF)
    print("train_corpus_tf_icf done")
    
    TF=term_frequency_paper(X_test,food_vocabulary_list)
    ICF=inverse_class_frequency(pos_corpus,neg_corpus,food_vocabulary_list)
    test_corpus_tf_icf=term_frequency_inverse_class_frequency(TF,ICF)
    print("test_corpus_tf_icf done")

    model1 = LinearSVC()
    model2 = MultinomialNB()   
    model3 = LogisticRegression()
    model1.fit(train_corpus_tf_icf,y_train)
    model2.fit(train_corpus_tf_icf,y_train)
    model3.fit(train_corpus_tf_icf,y_train)
    result1 = model1.predict(test_corpus_tf_icf)
    result2 = model2.predict(test_corpus_tf_icf)
    result3 = model3.predict(test_corpus_tf_icf)
    
     
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
