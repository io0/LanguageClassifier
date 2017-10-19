# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 10:58:16 2017

@author: Christopher
"""

import nltk
from collections import Counter
import string
import numpy as np
from operator import mul
from math import log
import matplotlib.pyplot as plt
import random

class mainParser():
    def __init__(self):
        self.targets = []
        self.targetCount = [0,0,0,0,0]
        self.charCollection = [[],[],[],[],[]]          # Aggregation of chars in each language
        self.charCount = []                             # List containing dictionaries of char frequencies in each language
        self.totalCharCounts = []                       # Number of chars in each language
        self.charSet = [[],[],[],[],[]]                 # Aggregation of chars, hashed once per utterance, in each language
        self.charSetCount = []                          # List containing dictionaries of char frequences per utterance in each language
        self.charSetTotalCount = []
        self.sumIDF = []
    def getTargets(self):
        with open('train_set_y.csv', 'r', encoding='utf8') as file:
            #for line in file:
            print(file.readline())
            for line in file:
                self.targets.append(int(line[-2:-1]))

    def collectChars(self):
        with open('train_set_x.csv', 'r', encoding='utf8') as file:
            print(file.readline())
            #i = 0
            for i in range(0,270000):
                line = file.readline()
                #for line in file:
                index = self.targets[i]
                utterance = "".join(line.lower().split()[1:-1])
                self.charCollection[index].extend(list(utterance))
                self.charSet[index].extend(set(utterance))
                self.targetCount[index] += 1
                #i = i + 1
    def countChars(self):
        for d in self.charCollection:
            count = Counter(d)
            '''for token in count.copy():
                if not token.isalnum():
                    count.pop(token)
            
            for token in count.copy():
                if token in "0123456789":
                    count.pop(token)
            
            for token in count:
                if token in string.punctuation:
                    count.pop(token)
            '''
            self.charCount.append(count)
            
            self.totalCharCounts.append(sum(count.values()))
        for d in self.charSet:
            count = Counter(d)
            self.charSetCount.append(count)
        self.charSetTotalCount = sum(self.charSetCount, Counter())
        for i in range (0, 5):
            sumTFIDF = 0
            for token in self.charSetCount[i]:
                tf = self.charSetCount[i][token]/self.totalCharCounts[i]
                idf = log(self.targetCount[i]/(self.charSetCount[i][token] + 1))
                sumTFIDF += tf*idf
            self.sumIDF.append(sumTFIDF)
                
    def naiveBayes(self, line):
        probXInY = [[],[],[],[],[]]
        utterance = list("".join(line.split()[1:-1]))
        totalUniqueChars = [len(d) for d in self.charCount]
        #random.shuffle(utterance)
        utterance = (Counter(utterance[0:20]))
        numChars = sum(utterance.values())
        for token in utterance:
            #tf = utterance[token]/numChars
            #tfidf = utterance[token]/numChars * log(270000/self.charSetCount[token])
            for i in range(0,5):
                #tfidf = tf*log(self.targetCount[i]/(self.charSetCount[i][token] + 1)
                #probXInY[i].append(log((tfidf)/(self.sumIDF[i] + totalUniqueChars[i])))
                probXInY[i].append(utterance[token] * log((self.charCount[i][token] + 1)/(self.totalCharCounts[i] + totalUniqueChars[i])))
                #probXInY[i].append(utterance[token] * log((self.charCount[i][token]/self.totalCharCounts[i] + 1)/(self.totalCharCounts[i] + totalUniqueChars[i])))
        probY = [sum(probabilities) for probabilities in probXInY]
        totalChars = sum(self.totalCharCounts)
        for i in range(0, 5):
            probY[i] = probY[i] + log(self.totalCharCounts[i] / totalChars)
        return probY
    
a = mainParser()
a.getTargets()
a.collectChars()
a.countChars()

predictions = []
'''
with open('test_set_x.csv', 'r', encoding='utf8') as file:
    file.readline()
    for line in file:
        prediction = np.argmax(a.naiveBayes(line))
        predictions.append(prediction)

with open('second_attempt.csv', 'w', encoding='utf8') as f2:
    print("Id,Category", file=f2)
    i = 0
    for n in predictions:
        s = str(i) + ',' + str(n)
        print(s, file=f2)
        i = i + 1
'''
with open('train_set_x.csv', 'r', encoding='utf8') as file:
    file.readline()
    #for line in file:
    for i in range(0,270000):
        line = file.readline()
    for i in range(0,6500):
        line = file.readline()
        prediction = np.argmax(a.naiveBayes(line.lower()))
        predictions.append(prediction)

'''
with open('second_attempt.csv', 'w', encoding='utf8') as f2:
    print("Id,Category", file=f2)
    i = 0
    for n in predictions:
        s = str(i) + ',' + str(n)
        print(s, file=f2)
        i = i + 1
[0.7679324894514767,
 0.897116134060795,
 0.4897959183673469,
 0.7776073619631901,
 0.7084870848708487]
        
numRight = [0,0,0,0,0]
numWrong = [0,0,0,0,0]
other = []
nt = 0
for i in range(0,5000):
    if predictions[i] == a.targets[270000 + i]:
        nt = nt + 1
        numRight[a.targets[270000 + i]] += 1
    else:
        numWrong[a.targets[270000 + i]] += 1
        if a.targets[270000 + i] == 2:
            other.append(predictions[i])
print(nt)
[numRight[i]/(numWrong[i] + numRight[i]) for i in range(0,5)]
'''