#!/home/gaurav/anaconda/bin/python

import csv 
import sys
import sklearn
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import cross_val_score


train_data = []
train_target = []
csvReader =open('./train_new.txt', 'rb')
v = CountVectorizer(max_features=100, stop_words='english')
for row in csvReader:  
	splits = row.split(",")  
#	print splits[2]                                                                        
	train_data.append(splits[2])
	train_target.append(splits[0])
#for i in range(0, len(train_data)):                                                                               
#	print train_data[i]
text_features = v.fit_transform(train_data)
#print v.get_feature_names()[1:50]

model = MultinomialNB()
cross_val_score(model, text_features,train_target)
cross_val_score(linear_model.LogisticRegression(), text_features, train_target, cv=5)


