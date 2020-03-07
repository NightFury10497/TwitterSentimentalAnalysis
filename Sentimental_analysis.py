import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import csv
import os,sys
from sklearn.feature_extraction.text import CountVectorizer
from time import time

from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer

f1 = open('stopwords.txt', 'r')
stoplist = []
for line in f1:
    nextlist = line.replace('\n', ' ').split()
    stoplist.extend(nextlist)
f1.close()
vectorizer =CountVectorizer(stop_words = stoplist, min_df=1)

def getVectorizer(tweets):
        X = vectorizer.fit_transform(tweets)
        tfidf_transformer = TfidfTransformer().fit(X)
        X2 = tfidf_transformer.transform(X)
        print(X.toarray().shape)
        print(X2.toarray().shape)
        return X

df1=pd.read_csv('train.csv',error_bad_lines=False)
df2=pd.read_csv('test.csv',error_bad_lines=False)
X=df1['Tweets']##tweets
y=df1['Polarity']##sents
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


#from sklearn.preprocessing import LabelEncoder
#labelencoder_X = LabelEncoder()
#X = labelencoder_X.fit_transform(X).reshape((-1, 1))


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state=42)
t1 = time()
X=getVectorizer(df1['Tweets'])
clf=svm.SVC()
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
try:

clf.fit(X.toarray(),df1['Polarity'])
test=y_train
t2=vectorizer.transform(X_test)
print('Prediction ')
y_pred=clf.predict(t2.toarray())
print(clf.score(X.toarray(),df1['Polarity']))




    labelencoder_y = LabelEncoder()
    y_pred = labelencoder_y.fit_transform(y_pred)

    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))  
    print(accuracy_score(y_test, y_pred))  

    ##trained Tweets and, Trained Sents
##print accuracy_score(X_test,y_test)
# print clf.score(X_test,y_test)
# print 'Accuracy of the TRAINED LABELS v/s PREDICTED LABELS USING CLASSIFIER'
# print ':--'
# print accuracy_score(y_test,y_pred)
# print ']]'
# print clf.score(y_test,y_pred)
# print '::'
except :
pass

print ("predicting time:" , round(time() - t1, 3), 's')

print ('Program finished')
