import re
import string
import operator
import numpy as np
import os
import logging
import pickle
from stemming.porter2 import stem
from unidecode import unidecode
from nltk import word_tokenize, sent_tokenize, ne_chunk, pos_tag
from nltk import pos_tag_sents
from nltk.chunk.regexp import RegexpParser
from nltk.chunk import tree2conlltags
from nltk.corpus import stopwords
from itertools import chain, groupby
from operator import itemgetter
from sklearn.feature_extraction.text import TfidfVectorizer

from tfidf import punct_re, stop_words


def get_list_of_texts(path):
    #path = "/Users/rossabramson/Documents/Junior NYU/NLP/Project/Data_Sources/SemEval2010/test"
    list_of_documents = []
    list_of_titles=[]
    for filename in os.listdir(path):
        list_of_titles.append(str(filename))
        print("Reading: "+str(filename))
        data=''
        with open("/Users/rossabramson/PycharmProjects/NLPFinal/bbc/" + filename,'r') as myfile:
            data += myfile.read().replace('\n', ' ')
        list_of_documents.append(data)


    return list_of_documents, list_of_titles

list_of_documents, titles=get_list_of_texts("/Users/rossabramson/PycharmProjects/NLPFinal/bbc/")
print("Begin\n")
i=0
with open("bbc_stemmed.txt",'w') as f:
    for doc in list_of_documents:

        for sentence in doc:
            # sentence = punct_re.sub(' ', sentence)  # remove punctuation

            for word in sentence:
                 f.write(str(stem(word)).lower())


        print("DOC--COMPLETE%d\n"%i)
        i+=1

