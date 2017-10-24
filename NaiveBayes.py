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

# Load data
df_x = pd.read_csv('train_set_x.csv', index_col='Id').values
df_y = pd.read_csv('train_set_y.csv', index_col='Id').values
valid = pd.read_csv('test_set_x.csv', index_col='Id').values

# Process
df_x[pd.isnull(df_x)] = '' #potentially uses tab chars instead
valid[pd.isnull(valid)] = ''
df = np.concatenate((df_x,df_y), axis=1)

for i,s in enumerate(df_x):
    df_x[i][0] = re.sub(r'http\S*', '' ,s[0] , re.UNICODE)
    df_x[i][0] = re.sub(r'[0-9]', '' ,s[0] , re.UNICODE)
    
# Data Splitting
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=14)
x_test_gen = [[] for i in range(len(x_test))]
for i, test_samples in enumerate(x_test):
    sequence = list(re.sub(r'\s', '', test_samples[0]))
    #shuffle(sequence)
    if len(sequence) > 20:
        sequence = sequence[:20]
    sequence = ''.join(sequence)
    x_test_gen[i] = sequence
x_test_gen = np.array(x_test_gen)

x_train_gen = [[] for i in range(len(x_train))]
for i, test_samples in enumerate(x_train):
    sequence = list(re.sub(r'\s', '', test_samples[0]))
    shuffle(sequence)
    sequence = ''.join(sequence)
    x_train_gen[i] = sequence
    
# Naive Bayes - Multinomial
pipeline_nb_m = Pipeline([
    ('vect', CountVectorizer(analyzer='char', lowercase=False)),
    ('clf', MultinomialNB()),
])
parameters_nb_m = {
    'vect__max_features': (1000, None)
}
t_nb_m = time()
gs_nb = GridSearchCV(pipeline_nb_m,parameters_nb_m, n_jobs=-1)
gs_nb = gs_nb.fit(x_train_gen,y_train.flatten().tolist())
print ('Done %03f' % (time() - t_nb_m))
pred_nb = gs_nb.predict(x_test_gen.flatten().tolist())
print ('Naive Bayes %02.02f%%' % (np.mean(pred_nb == y_test.flatten().tolist())*100))
for param_name in sorted(parameters_nb_m.keys()):
    print ('%s: %r' % (param_name, gs_nb.best_params_[param_name]))
    
# Saving predictions to an output file for submission to the competition
filename = 'io0-predictions-nb.csv'
predictions = gs_nb.predict(valid.flatten().tolist())
with open(filename,'w') as out:
    out.write('Id,Category\n')
    for i, e in enumerate(predictions):
        out.write(str(i))
        out.write(',')
        out.write(str(e))
        out.write('\n')