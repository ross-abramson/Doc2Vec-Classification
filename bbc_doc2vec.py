
import matplotlib

import pylab
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np
import sys
import gensim
import numpy as np
import math
import re
import random
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from pprint import pprint
import nltk
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from os import listdir
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix



from pandas import DataFrame as df
from os.path import isfile, join

PERCENT = 0.9

#now create a list that contains the name of all the text file in your data #folder
class LabeledLineSentence(object):
    def __init__(self, doc_list, labels_list):
        self.labels_list = labels_list
        self.doc_list = doc_list

    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
            s=[]


            for wor in doc:
                s.append(wor)
            yield gensim.models.doc2vec.TaggedDocument(s, [self.labels_list[idx]])

def nlp_clean(data):
    tokenizer = RegexpTokenizer(r'\w+')
    stopword_set = set(stopwords.words('english'))
    new_data = []

    for d in data:
        new_str = d.lower()

        new_str = re.sub('^[0-9]+', '', new_str) #Removes numbers
        dlist = tokenizer.tokenize(new_str)
        dlist = list(set(dlist).difference(stopword_set))

        new_data.append(dlist)
    return new_data

def convert_labels_to_topics(docLabels):
    new_doc_labs=[]
    for i, doc in enumerate(docLabels):
        if doc[3] == 'b':
            new_doc_labs.append("b")
        elif doc[3] == 't':
            new_doc_labs.append("t")
        elif doc[3] == 's':
            new_doc_labs.append("s")
        elif doc[3] == 'e':
            new_doc_labs.append("e")
        elif doc[3] == 'p':
            new_doc_labs.append("p")
        else:
            print("ERROR")
    return new_doc_labs

def get_train_test_labels(convert_to_topics=False):
    #Returns test, train, and labels list
    docLabels = [f for f in listdir("bbc") if
                 f.endswith('.txt')]
    # create a list data that stores the content of all text files in order of their names in docLabels
    data = []
    for doc in docLabels:
        data.append(open("bbc/" + doc).read().replace('\n', ' '))
    print("DATA COLLECTED")

    if convert_to_topics == True:
        new_labels=convert_labels_to_topics(docLabels)
        data = nlp_clean(data)
        train, test = split_data(data, PERCENT)
        print("LABELS COLLECTED")
        return train, test, new_labels
    else:
        data = nlp_clean(data)
        train, test = split_data(data, PERCENT)
        print("LABELS COLLECTED")
        return train, test, docLabels


    # This function does all cleaning of data using two objects above

def split_data(data, percent):

    num_of_train=math.floor(len(data)*percent)
    train_data=[]
    test_data=[]
    for i in range(0,num_of_train):
        train_data.append(data[i])
    for i in range(num_of_train, len(data)):
        test_data.append(data[i])



    return train_data, test_data

def training(train_data,docLabels):

    it = LabeledLineSentence(train_data, docLabels)
    print("TRAINING MODEL")
    model = gensim.models.Doc2Vec( documents=it, vector_size=100, window=10, epochs=100, min_count=1, workers=4, alpha=0.025, min_alpha=0.025)
    model.save("bbc-doc2vec-name-dbow-both.model")

def kmeans_plot(model):

    # create and apply PCA transform
    X=model.docvecs.doctag_syn0
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X)

    kmeans = KMeans(n_clusters=5, max_iter=100)
    kmeans.fit(principal_components)
    y_kmeans = kmeans.predict(principal_components)
    # plot data with seaborn
    plt.scatter(principal_components[:, 0], principal_components[:, 1], c=y_kmeans, s=50, cmap='viridis')
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)

def char_plot(model, labels):
    char_labels=labels
    if len(labels[0]) > 1:
        char_labels=convert_labels_to_topics(labels)
    print("Loading char model...")
    X = model.docvecs.doctag_syn0
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X)

    # slice the 2D array
    x = principal_components[:, 0]
    y = principal_components[:, 1]

    # plot with text annotation
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.scatter(x, y, s=0)


    for i, label in enumerate(char_labels):
        ax.annotate(label, (x[i], y[i]), size='x-large')


    print("Char model loaded...")

def knn_accuracy_plot(model):
    number_of_k=120
    iterations = 1000
    accuracy = [0]*number_of_k

    char_labels = convert_labels_to_topics(labels)
    for j in range (0,iterations):
        X = model.docvecs.doctag_syn0
        # pca = PCA(n_components=2)
        # principal_components = pca.fit_transform(X)

        np_X = np.asarray(X)
        np_Y = np.asarray(char_labels)
        s = np.arange(np_X.shape[0])
        np.random.shuffle(s)
        coordinates = np_X[s]
        tags = np_Y[s]

        train = math.floor(len(X) * PERCENT)

        x_train = coordinates[:train]
        x_test = coordinates[train:]
        y_train = tags[:train]
        y_test = tags[train:]

        # Calculating error for K values between 1 and 40
        for i in range(1, number_of_k+1):
            knn = KNeighborsClassifier(n_neighbors=i)
            knn.fit(x_train, y_train)
            y_pred = knn.predict(x_test)
            accuracy[i-1]+=(accuracy_score(y_test, y_pred))
    for i in range (0,number_of_k):
        accuracy[i]/=iterations
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, number_of_k+1), accuracy, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)
    plt.title('Accuracy Rate K Value')
    plt.xlabel('K Value')
    plt.ylabel('Accuracy')

def knn_error_plot(x_train, y_train, x_test, y_test):
    error = []

    # Calculating error for K values between 1 and 40
    for i in range(1, 51):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(x_train, y_train)
        pred_i = knn.predict(x_test)
        error.append(np.mean(pred_i != y_test))
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, 51), error, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)
    plt.title('Error Rate K Value')
    plt.xlabel('K Value')
    plt.ylabel('Mean Error')

def knn(model, labels, k, acc):
    char_labels=convert_labels_to_topics(labels)
    X = model.docvecs.doctag_syn0
    # pca = PCA(n_components=2)
    # principal_components = pca.fit_transform(X)

    np_X = np.asarray(X)
    np_Y = np.asarray(char_labels)
    s = np.arange(np_X.shape[0])
    np.random.shuffle(s)
    coordinates=np_X[s]
    tags=np_Y[s]

    train = math.floor(len(X) * PERCENT)


    x_train = coordinates[:train]
    x_test = coordinates[train:]
    y_train = tags[:train]
    y_test = tags[train:]


    # instantiate learning model for a given k
    knn = KNeighborsClassifier(n_neighbors=k)

    # fitting the model
    knn.fit(x_train, y_train)
    # predict the response
    y_pred = knn.predict(x_test)
    # evaluate accuracy


    print("K="+str(k)+", accuracy score: "+str(accuracy_score(y_test, y_pred)))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    if (acc==True):
        knn_accuracy_plot(model)

if __name__=='__main__':

    args = sys.argv[1:]
    acc = False
    if (len(args)!=3):
        print("ERROR pass K value, a model viewer (-k for knn and -m for kmeans, and -y or -n to retrain")
        print("Ex: 'bb_doc2vec.py 35 -k -n")
        sys.exit(1)
    k=int(args[0])
    plot = str(args[1])
    train = str(args[2])
    if (len(args) == 4 and args[3]=="-a"):
        acc = True

    print("Getting training, testing, and labeled data")
    train,test,labels=get_train_test_labels()
    if train=="-y":
        print("Training data")
        training(train_data=(train+test),docLabels=labels)
    print("Loading model")
    d2v_model = gensim.models.doc2vec.Doc2Vec.load('bbc-doc2vec-name-dbow-both.model')
    if(plot == "-k"):
        knn(d2v_model, labels, k, acc)
    if(plot == "-m"):
        kmeans_plot(d2v_model)
    else:
        print("DID NOT INPUT SECOND ARG AS '-k' FOR KNN OR -m FOR KMEANS")
        print("Ex: 'bb_doc2vec.py 35 -k -n")
    char_plot(d2v_model,labels) #2d Principal Component Analysis

    pylab.show()

    print("DONE")


####
#https://stackoverflow.com/questions/47930809/doc2vec-clustering-resulting-vectors
#https://github.com/stefanpernes/word-embedding/blob/master/doc2vec.py