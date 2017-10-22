# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 00:13:38 2017

@author: Marley
"""
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
from sklearn import preprocessing
neigh = KNeighborsClassifier(n_neighbors=3)
targets = []
def getTargets():
        with open('train_set_y.csv', 'r', encoding='utf8') as file:
            print(file.readline())
            for line in file:
                targets.append(int(line[-2:-1]))
allChars = []
def collectChars():
    with open('train_set_x.csv', 'r', encoding='utf8') as file:
        print(file.readline())
        #i = 0
        for i in range(0,270000):
            line = file.readline()
            utterance = "".join(line.lower().split()[1:-1])
            allChars.extend(list(utterance))
getTargets()
collectChars()
allChars = list(set(allChars))
dims = len(allChars)
endTargets = []
X = []
with open('train_set_x.csv', 'r', encoding='utf8') as file:
            print(file.readline())
            #i = 0
            for i in range(0,270000):
                line = file.readline()
                if (len(line) > 19):
                    vec = [0 for i in range(0, dims)]
                    utterance = "".join(line.lower().split()[1:-1])
                    utterance = (Counter(utterance[]))
                    for token in utterance:
                        if token in allChars:
                            vec[allChars.index(token)] = utterance[token]
                    X.append(vec)
                    endTargets.append(targets[i])
X = preprocessing.normalize(X, norm='l2')
neigh.fit(X[0:50000],endTargets[0:50000])
predictions = []
with open('train_set_x.csv', 'r', encoding='utf8') as file:
    file.readline()
    #for line in file:
    for i in range(0,270000):
        line = file.readline()
    for i in range(0,1000):
        line = file.readline()
        vec = [0 for i in range(0, dims)]
        utterance = "".join(line.lower().split()[1:-1])
        utterance = (Counter(utterance))
        for token in utterance:
            if token in allChars:
                vec[allChars.index(token)] = utterance[token]
        vec = preprocessing.normalize(vec, norm='l2')
        predictions.append(neigh.predict(vec))
        

nt = 0
numRight = [0,0,0,0,0]
numWrong = [0,0,0,0,0]
for i in range(0,1000):
    if int(predictions[i]) == targets[270000 + i]:
        nt = nt + 1
        numRight[targets[270000 + i]] += 1
    else:
        numWrong[targets[270000 + i]] += 1
        #if targets[270000 + i] == 2:
        #    other.append(predictions[i])
print(nt)
print(numRight)
print(numWrong)
print([numRight[i]/(numRight[i] + numWrong[i]) for i in range(0,5)])
trial.append(numRight)
trial.append(numWrong)
'''
train 50000, k neighbors 2-9
[0.58, 0.8107074569789675, 0.502092050209205, 0.5174825174825175, 0.4]                 2
[0.58, 0.7762906309751434, 0.6234309623430963, 0.6503496503496503, 0.5111111111111111] 3
[0.56, 0.8011472275334608, 0.5606694560669456, 0.6013986013986014, 0.5333333333333333] 4
[0.54, 0.7915869980879541, 0.6485355648535565, 0.6223776223776224, 0.5111111111111111] 5
[0.56, 0.8202676864244742, 0.6108786610878661, 0.6503496503496503, 0.6222222222222222] 7
[0.58, 0.8202676864244742, 0.5732217573221757, 0.6363636363636364, 0.6]                8
[0.54, 0.8126195028680688, 0.5899581589958159, 0.6503496503496503, 0.5555555555555556] 9

0.665
0.7
0.691
0.708
0.724
0.713
0.711
'''