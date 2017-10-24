import csv
import re
import math
import sys
import operator
import string
from random import shuffle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from time import time
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import euclidean_distances

# Euclidian Distance calculation function
def EuclidianDis(point1, point2, length):
    dist = 0
    for x in range(length):
        dist += pow((point1[x] - point2[x]), 2)
    return math.sqrt(dist)

# The function for getting the K neighbors  
def GetNeighbors(data, labels, test_sample, k):
    distances = []
    for i, train_sample in enumerate(data):
        sys.stdout.write('%06d\r' % i)
        sys.stdout.flush()
        interval = euclidean_distances(train_sample, test_sample)
        distances.append((train_sample, interval))
    neighbors = [(x[0], x[1], y) for x, y in zip(distances, labels)]
    neighbors = [(i,j,k) for (i,j,k) in sorted(neighbors, key=operator.itemgetter(1), reverse=True)]
    return neighbors[:k]
    
# The prediction response based on neighbors
def PredictedResponse(neighbors):
    Votes = dict()
    for i,x in enumerate(neighbors):
        res = x[-1]
        if res in Votes:
            Votes[res] += 1
        else:
            Votes[res] = 1
    return max(Votes.items(), key=operator.itemgetter(1))[0]

# The accuracy of obtained predictions
def GetAccuracy(test_labels, predictions):
    correct = 0
    for i,y in enumerate(test_labels):
        if y == predictions[i]:
            correct += 1
    return (correct/float(len(test_labels))) * 100.00

# Load data
df_x = pd.read_csv('train_set_x.csv', index_col='Id').values
df_y = pd.read_csv('train_set_y.csv', index_col='Id').values
valid = pd.read_csv('test_set_x.csv', index_col='Id').values

# PreProcess
df_x[pd.isnull(df_x)] = '' #potentially uses tab chars instead
valid[pd.isnull(valid)] = ''

for i,s in enumerate(df_x):
    df_x[i][0] = re.sub(r'http\S*', '' ,s[0] , re.UNICODE)
    df_x[i][0] = re.sub(r'[0-9]', '' ,s[0] , re.UNICODE)
    
df_x = df_x.flatten().tolist()
valid = valid.flatten().tolist()
    
# K Nearest Neighbors
v = CountVectorizer(analyzer='char', max_features=263)
t_knn = v.fit_transform(df_x)
v_knn = v.fit_transform(valid)

# K-Nearest Neighbors
pred_knn = []
k = 5
for i, s in enumerate(v_knn):
    sys.stdout.write('000000 %06d\r' % i)
    sys.stdout.flush()
    
    neighbors = GetNeighbors(t_knn, df_y.flatten().tolist(), s, k)
    results = PredictedResponse(neighbors)
    pred_knn.append(results)
    
# Saving predictions to an output file for submission to the competition
filename = 'io0-predictions-knn.csv'
with open(filename,'w') as out:
    out.write('Id,Category\n')
    for i, e in enumerate(pred_knn):
        out.write(str(i))
        out.write(',')
        out.write(str(e))
        out.write('\n')