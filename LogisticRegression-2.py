import csv
import sys
import sklearn
from sklearn import linear_model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import cross_val_score
import numpy as np
from sklearn import metrics
from  sklearn.metrics  import auc_score
from sklearn.linear_model import LogisticRegression

train_data = []
train_target = []
test_data = []
test_ids =[]

## clean up the train file before use

##sed 's/\\n//g'  train.csv > train1.csv
## python decode.py train1.csv train_new.txt
csvReader =open('./train_new.txt', 'rb')
v = CountVectorizer(ngram_range=(1, 3),strip_accents='unicode' , stop_words='english')

## read training data
for row in csvReader:
        splits = row.split(",")
        train_data.append(splits[2])
        train_target.append(int(splits[0]))

train_text_features = v.fit_transform(train_data)
train_target = np.array(train_target)
model1 = MultinomialNB()
model2 = LogisticRegression()

## compare the models ...

scores1 = cross_val_score(model1, train_text_features, train_target,  score_func=auc_score, cv=5)
scores2 = cross_val_score(model2, train_text_features, train_target,  score_func=auc_score, cv=5)
print np.mean(scores1) ## 75 %
print np.mean(scores2) ## 72 %

classifier = model1.fit(train_text_features, train_target)

## clean up the test file before use

##sed 's/\\n//g'  test.csv > test_newer.csv
## python decode.py test_newer.csv test_newest.txt

tgtReader=open('./test_newest.txt', 'rb')
for tgt in tgtReader:
        splits = tgt.split(",")
        test_data.append(splits[2])
	test_ids.append(int(splits[0]))


# Evaluate the classifier on the testing set
X_test = v.transform(test_data)
result=  classifier.predict_proba(X_test)

## write result in csv

#w = csv.writer(open('./ouput.csv','w'))
w = open('./ouput.csv','w')
w.write('id,insult')
for i in range(len(result)):
    print test_ids[i],result[i][1]

