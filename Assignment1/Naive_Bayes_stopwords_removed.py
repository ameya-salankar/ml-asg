# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 20:34:29 2020

@author: Ayush RKL
"""

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import KFold

df = pd.read_csv("./a1_data/a1_d3.txt", header = None, delimiter = '\t')

stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
          
#print(np.array_split(df,5))
#print(type(np.array_split(df, 5)))
#print(np.array_split(df, 5)[0].shape)
#a = np.array_split(df, 5)[0]
#print(type(a))

ans = np.zeros(5)
i = 0

print("************Accuracy for 5 folds*************")
kf = KFold(5);
for train, test in kf.split(df):
    df_train = df.iloc[train]
    df_test = df.iloc[test]
#    print(df_test)
    bag = {}
    pos_bag = {}
    neg_bag = {}
    for index, row in df_train.iterrows():
        s = re.sub("[^a-zA-Z]+", " ", row[0]).lower()
        words = s.split()
        if row[1]==1:
            for e in words:
                if e in stop_words:
                    continue
                if e in pos_bag:
                    bag[e] += 1
                    pos_bag[e] += 1
                else:
                    pos_bag[e] = 1
                    if e not in bag:
                        bag[e] = 1
                    else:
                        bag[e] += 1
                
        else:
            for e in words:
                if e in stop_words:
                    continue
                if e in neg_bag:
                    bag[e] += 1
                    neg_bag[e] += 1
                else:
                    neg_bag[e] = 1
                    if e not in bag:
                        bag[e] = 1
                    else:
                        bag[e] += 1
    
    bag_sz = len(bag)
    pbag_sz = len(pos_bag)
    nbag_sz = len(neg_bag)
#    print(vocab_sz)
    
    bag_sum = sum(bag.values())
    pbag_sum = sum(pos_bag.values())
    nbag_sum = sum(neg_bag.values())
    
    count = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    
    
    for index, row in df_test.iterrows():
        wrds = re.sub("[^a-zA-Z]+", " ", row[0]).lower().split()
#        print(wrds)
        t_p = 1 #total positive probability
        t_n = 1 #total negative probability
        
        for word in wrds:
            if word in stop_words:
                continue
            p = ((pos_bag[word] if word in pos_bag else 0) + 1)/(pbag_sum + pbag_sz + 1)
            n = ((neg_bag[word] if word in neg_bag else 0) + 1)/(nbag_sum + nbag_sz + 1)
#            print(p, n)
            t_p = t_p * p
            t_n = t_n * n
        
#        print(t_p, t_n, "\n")
        if t_p > t_n and row[1] == 1:
            count += 1
            tp += 1
        elif t_p < t_n and row[1] == 0:
            count += 1
            tn += 1
        elif t_p > t_n and row[1] == 0:
            fp += 1
        elif t_p < t_n and row[1] == 1:
            fn += 1
            
    acc = count/(len(df_test))
    print("Accuracy =", acc)
    ans[i] = acc
    i += 1
    precision = tp/(tp + fp)
    recall = tp/(tp + fn)
    fscore = (2*precision*recall)/(precision + recall)
    print("F-Score =", fscore, "\n")

print("*************Overall Accuracy****************")
print("Overall Accuracy = ", np.mean(ans), "+/-", np.std(ans))
            
    