import numpy as np
import cPickle
import rnn as rnned
from util import * 
import sys
import random

model_path = '../best.mdl'
dict_path = '../data/dict.pkl'
train_path = '../data/train.pkl'

print 'load model...'
best = cPickle.load(open(model_path, "r"))
rParameters = best[0]

#load data
word_to_index = cPickle.load(open(dict_path, "r"))
index_to_word = {value:key for key, value in word_to_index.items()}
train = cPickle.load(open(train_path, 'r'))

#parameters == train.py

nhidden = 512
vobsize = len(word_to_index)
emb_dimension = 200

rnn = rnned.RNNED(nh=nhidden, nc=vobsize, de=emb_dimension, model= rParameters)

while True:
    wait = raw_input("please press enter")
    i = random.randint(1, len(train)-1)
    x_train = train[i][1]
    sentence = [index_to_word[x] for x in x_train[1:]]
    print " ".join(sentence)
    c = train[i][0]
    generate_sentence(rnn, index_to_word, word_to_index, c)
        


'''
for i in range(20):
    c = train[i][0]
    generate_sentence(rnn, index_to_word, word_to_index, c)


'''
