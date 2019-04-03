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

###Inverse Document Frequency###
def inverse_document_frequency(documents,vocabulary_list):
    IDF=[]
    N=len(documents)
    for word in vocabulary_list:
        count=0
        for document in documents:
            if word in document:
                count+=1
        if count == 0:
            idf=0
        else:
            idf=math.log(N/count,2)
        IDF.append(idf)
    return IDF


###Term Frequency Inverse Document Frequency###
def term_frequency_inverse_document_frequency(TF,IDF):
	IDF=np.array(IDF)
	TFIDF=[]
	for tf in TF:
		tf=np.array(tf)
		tfidf=tf*IDF
		TFIDF.append(tfidf.tolist())
	return TFIDF


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
    IDF=inverse_document_frequency(X_train,food_vocabulary_list)
    train_corpus_tf_idf=term_frequency_inverse_document_frequency(TF,IDF)
    print("train_corpus_tf_idf done")

    TF=term_frequency_paper(X_test,food_vocabulary_list)
    IDF=inverse_document_frequency(X_test,food_vocabulary_list)
    test_corpus_tf_idf=term_frequency_inverse_document_frequency(TF,IDF)
    print("test_corpus_tf_idf done")


    
    model1 = LinearSVC()
    model2 = MultinomialNB()   
    model3 = LogisticRegression()
    model1.fit(train_corpus_tf_idf,y_train)
    model2.fit(train_corpus_tf_idf,y_train)
    model3.fit(train_corpus_tf_idf,y_train)
    result1 = model1.predict(test_corpus_tf_idf)
    result2 = model2.predict(test_corpus_tf_idf)
    result3 = model3.predict(test_corpus_tf_idf)
    
     
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