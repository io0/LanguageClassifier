# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 10:58:16 2017

@author: Christopher
"""

from collections import Counter
import string
import numpy as np
from operator import mul
from math import log
import matplotlib.pyplot as plt
from random import shuffle
import itertools
import re


class NaiveBayes():
    def __init__(self, lines, targets, ngrams = 1):
        self.lines = lines
        self.targets = targets
        self.ngrams = ngrams
        self.targetCount = [0,0,0,0,0]
        self.charCollection = [[],[],[],[],[]]          # Aggregation of chars in each language
        self.charCount = []                             # List containing dictionaries of char frequencies in each language
        self.totalCharCounts = []                       # Number of chars in each language
        self.totalUniqueChars = 0                       # Number of unique chars in vocabulary
        
        self.charSet = [[],[],[],[],[]]                 # Aggregation of chars, hashed once per utterance, in each language
        self.charSetCount = []                          # List containing dictionaries of char frequences per utterance in each language
        self.charSetTotalCount = []
        self.sumTFIDF = []
        
        self.collectChars()
        self.countChars()

    def collectChars(self):
        i = 0
        for utterance in self.lines:
            index = self.targets[i]
            self.charCollection[index].extend(self.getGrams(utterance))
            self.charSet[index].extend(set(utterance))
            self.targetCount[index] += 1
            i += 1
    def countChars(self):
        for d in self.charCollection:
            count = Counter(d)
            '''for token in count.copy():                # Some data cleanup; allocated elsewhere
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
        self.totalUniqueChars = len(sum(self.charCount, Counter()))
        '''
        for d in self.charSet:
            count = Counter(d)
            self.charSetCount.append(count)
        self.charSetTotalCount = sum(self.charSetCount, Counter())
        for i in range (0, 5):
            sumtfidf = 0
            for token in self.charSetCount[i]:
                tf = self.charSetCount[i][token]/self.totalCharCounts[i]
                idf = log(self.targetCount[i]/(self.charSetCount[i][token] + 1))
                sumtfidf += tf*idf
            self.sumTFIDF.append(sumtfidf)
        '''
    def getGrams(self, utterance, randomize=False):
        chars = list(utterance)
        if randomize:
            shuffle(chars)
        grams = []#chars
        for i in range (0, len(chars) - self.ngrams + 1):
            gram = ''
            for j in range (0, self.ngrams):
                gram += chars[i + j]
            grams.append(gram)
        #grams = itertools.combinations(chars, 2)
        return grams
        
    def predict(self, line):
        probXInY = [[],[],[],[],[]]
        utterance = self.getGrams(line, randomize=True)
        utterance = (Counter(utterance))
        #numChars = sum(utterance.values())
        for token in utterance:
            #tf = utterance[token]/numChars
            for i in range(0,5):
                #tfidf = tf*log(self.targetCount[i]/(self.charSetCount[i][token] + 1))
                #probXInY[i].append(log(tfidf/(sum(self.sumTFIDF) + self.totalUniqueChars)))
                probXInY[i].append(utterance[token] * log((self.charCount[i][token] + 1)/(self.totalCharCounts[i] + self.totalUniqueChars)))
                #probXInY[i].append(utterance[token] * log((self.charCount[i][token]/self.totalCharCounts[i] + 1)/(1 + self.totalUniqueChars)))
        probY = [sum(probabilities) for probabilities in probXInY]
        #totalChars = sum(self.totalCharCounts)
        for i in range(0, 5):
            probY[i] = probY[i] + log(self.targetCount[i] / sum(self.targetCount))
            #probY[i] = probY[i] + log(self.totalCharCounts[i] / totalChars)
        return np.argmax(probY)

def findAccuracy(predictions, targets):
    numRight = [0,0,0,0,0]
    numWrong = [0,0,0,0,0]
    nt = 0
    for i in range(0,len(predictions)):
        if predictions[i] == targets[i]:
            nt = nt + 1
            numRight[targets[i]] += 1
        else:
            numWrong[targets[i]] += 1
    print(nt/len(predictions))
    print([numRight[i]/(numWrong[i] + numRight[i]) for i in range(0,5)])
    
if __name__ == "__main__":
    targets = []
    lines = []
    num_samples = 0
    with open('train_set_y.csv', 'r', encoding='utf8') as file:
            #for line in file:
            print(file.readline())
            for line in file:
                targets.append(int(line[-2:-1]))
                num_samples += 1
    with open('train_set_x.csv', 'r', encoding='utf8') as file:
            print(file.readline())
            for line in file:
                line = "".join(line.split()[1:-1])            
                #line = re.sub(r'\W+', '', line)
                lines.append(line)
    partition = int(.1*num_samples)
    for i in range (0,10):                                                      # 10-fold cross-validation
        a = NaiveBayes(lines[0:9*partition], targets, ngrams = 1)
        predictions = []
        for line in lines[9*partition:]:
            prediction = a.predict(line[0:20])
            predictions.append(prediction)
        findAccuracy(predictions, targets[9*partition:])
        lines = lines[partition:] + lines[:partition]                           # Rotate list to find next
        targets = targets[partition:] + targets[:partition]
'''
with open('second_attempt.csv', 'w', encoding='utf8') as f2:
    print("Id,Category", file=f2)
    i = 0
    for n in predictions:
        s = str(i) + ',' + str(n)
        print(s, file=f2)
        i = i + 1
0.7590932099211801
[0.7429553264604811, 0.8931779898933183, 0.5065883700945288, 0.7377003869541183, 0.7247232472324723]
0.7626726444428376
[0.738030095759234, 0.8926634313445618, 0.5211048158640227, 0.7528936742934051, 0.7317246273953159]
0.7667582616241232
[0.738061797752809, 0.895497934178658, 0.5270212462569513, 0.763981803585764, 0.715076071922545]
0.7641911924217225
[0.7289586305278174, 0.8938488576449912, 0.520645806543792, 0.7548315847598012, 0.724113475177305]
0.7649143105069057
[0.7501831501831502, 0.8945879023700035, 0.5205171444631815, 0.7556278817466775, 0.7357933579335794]
0.7683129655072674
[0.7312056737588652, 0.8954372623574145, 0.5288888888888889, 0.7652785289345592, 0.7159504734158776]
0.7631426711982067
[0.7281410727406319, 0.8942950633091478, 0.5173002990175138, 0.7529956427015251, 0.7422279792746114]
0.7603948224745101
[0.7430997876857749, 0.8930485053581501, 0.5034742327735958, 0.7530201342281879, 0.7219887955182073]
0.7650589341239424
[0.739590684544813, 0.8908078387142253, 0.5168391893862079, 0.7674907943187795, 0.7259684361549498]
0.7627811121556151
[0.7371663244353183, 0.892872416250891, 0.5179254640782202, 0.7542735042735043, 0.7423133235724744]
 
'''