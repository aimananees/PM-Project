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

#Food Reviews Vocabulary
def make_Corpus(root_dir,polarity_dirs):
    corpus = []
    for polarity_dir in polarity_dirs:
        reviews = [os.path.join(polarity_dir,f) for f in os.listdir(polarity_dir)]
        for review in reviews:
            doc_string = "";
            with open(review) as rev:
                for line in rev:
                    #line = preprocessing(line)
                    doc_string = doc_string + line
                    doc_string+=" "
            if not corpus:
                corpus = [doc_string]
            else:
                corpus.append(doc_string)
    return corpus

root_dir = 'Food_Reviews/pos/'
pos_corpus = make_Corpus(root_dir,['Food_Reviews/pos/'])
print("Positive Corpus Successful")

root_dir = 'Food_Reviews/neg/'
neg_corpus = make_Corpus(root_dir,['Food_Reviews/neg/'])
print("Negative Corpus Successful")


"""
from random import shuffle
shuffle(pos_corpus)
shuffle(neg_corpus)

from sklearn.model_selection import train_test_split

pos_train, pos_test = train_test_split(pos_corpus, test_size=0.2)
neg_train, neg_test = train_test_split(pos_corpus, test_size=0.2)

y_pos_train=[1]*len(pos_train)
y_neg_train=[0]*len(neg_train)
y_pos_test=[1]*len(pos_test)
y_neg_test=[0]*len(neg_test)

y_train=y_pos_train+y_neg_train
y_test=y_pos_test+y_neg_test


review_training_corpus=pos_train+neg_train
review_testing_corpus=pos_test+neg_test

for i in range(len(review_training_corpus)):
        review_training_corpus[i] = review_training_corpus[i].split(" ")
f = open('food_training_corpus.txt', 'w')
simplejson.dump(review_training_corpus, f)
f.close()
        
for i in range(len(review_testing_corpus)):
        review_testing_corpus[i] = review_testing_corpus[i].split(" ")
f = open('food_testing_corpus.txt', 'w')
simplejson.dump(review_testing_corpus, f)
f.close()

for i in range(len(pos_train)):
        pos_train[i] = pos_train[i].split(" ")
f = open('food_pos_train.txt', 'w')
simplejson.dump(pos_train, f)
f.close()

for i in range(len(neg_train)):
        neg_train[i] = neg_train[i].split(" ")
f = open('food_neg_train.txt', 'w')
simplejson.dump(neg_train, f)
f.close()

for i in range(len(pos_test)):
        pos_test[i] = pos_test[i].split(" ")
f = open('food_pos_test.txt', 'w')
simplejson.dump(pos_test, f)
f.close()

for i in range(len(neg_test)):
        neg_test[i] = neg_test[i].split(" ")

f = open('food_neg_test.txt', 'w')
simplejson.dump(neg_test, f)
f.close()    

f = open('y_train.txt', 'w')
simplejson.dump(y_train, f)
f.close() 

f = open('y_test.txt', 'w')
simplejson.dump(y_test, f)
f.close() 
"""
corpus=pos_corpus+neg_corpus
for i in range(len(corpus)):
        corpus[i] = corpus[i].split(" ")
        
for i in range(len(pos_corpus)):
        pos_corpus[i] = pos_corpus[i].split(" ")

for i in range(len(neg_corpus)):
        neg_corpus[i] = neg_corpus[i].split(" ")

f = open('corpus.txt', 'w')
simplejson.dump(corpus, f)
f.close()    

f = open('pos_corpus.txt', 'w')
simplejson.dump(pos_corpus, f)
f.close() 

f = open('neg_corpus.txt', 'w')
simplejson.dump(neg_corpus, f)
f.close()  
def create_vocabulary(corpus):
    vocabulary=Counter()
    for i in range(len(corpus)):
        vocabulary.update(corpus[i])   
    vocabulary_list = [word for word,frequency in vocabulary.items() if frequency >= 100]
    print("Vocabulary Generated")
    
    return vocabulary_list

review_vocabulary_list=create_vocabulary(corpus)


print(len(review_vocabulary_list))
f = open('food_vocabulary_list.txt', 'w')
simplejson.dump(review_vocabulary_list, f)
f.close()

print("Successful")
