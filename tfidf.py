#adding in stemmer
#TF-IDF algorithm mainly sourced from here: http://bdewilde.github.io/blog/2014/09/23/intro-to-automatic-keyphrase-extraction/

import re
import string
import operator
import numpy as np
import sys
import os
import logging
import pickle
#from stemming.porter2 import stem
from unidecode import unidecode
from nltk import word_tokenize, sent_tokenize, ne_chunk, pos_tag
from nltk import pos_tag_sents
from nltk.chunk.regexp import RegexpParser
from nltk.chunk import tree2conlltags
from nltk.corpus import stopwords
from itertools import chain, groupby

from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger('keyword-extraction')
logger.setLevel(logging.DEBUG)
punct_re = re.compile('[{}]'.format(re.escape(string.punctuation)))
stop_words = set(stopwords.words('english'))
list_of_texts=[]
def generate_candidate(texts, method='word', remove_punctuation=False):
    """
    Generate word candidate from given string
    Parameters
    ----------
    texts: str, input text string
    method: str, method to extract candidate words, either 'word' or 'phrase'
    Returns
    -------
    candidates: list, list of candidate words
    """
    words_ = list()
    candidates = list()

    # tokenize texts to list of sentences of words
    sentences = sent_tokenize(texts)
    for sentence in sentences:
        if remove_punctuation:
            sentence = punct_re.sub(' ', sentence) # remove punctuation
        words = word_tokenize(sentence)
        #words = list(map(lambda s: s.lower(), words))

        #Adding in NER Chunking

        chunks = ne_chunk(pos_tag(words))
        tokenized_sentence = [w[0] if isinstance(w, tuple) else " ".join(t[0] for t in w) for w in chunks]
        tokenized_sentence = list(map(lambda s: s.lower(), tokenized_sentence))
        documents = []
        for word in tokenized_sentence:
            documents.append(word)

        words_.append(documents)
    tagged_words = pos_tag_sents(words_) # POS tagging

    if method == 'word':
        tags = set(['JJ', 'JJR', 'JJS', 'NN', 'NNP', 'NNS', 'NNPS'])
        tagged_words = chain.from_iterable(tagged_words)
        for word, tag in tagged_words:
            if tag in tags and word.lower() not in stop_words:
                candidates.append(word)
    elif method == 'phrase':
        grammar = r'KT: {(<JJ>* <NN.*>+ <IN>)? <JJ>* <NN.*>+}'
        chunker = RegexpParser(grammar)
        all_tag = chain.from_iterable([tree2conlltags(chunker.parse(tag)) for tag in tagged_words])
        for key, group in groupby(all_tag, lambda tag: tag[2] != 'O'):
            candidate = ' '.join([word for (word, pos, chunk) in group])
            if key is True and candidate not in stop_words:
                candidates.append(candidate)
    else:
        print("Use either 'word' or 'phrase' in method")
    return candidates

def keyphrase_extraction_tfidf(texts, method='word', min_df=5, max_df=0.8, num_key=int(sys.argv[1])):
    """
    Use tf-idf weighting to score key phrases in list of given texts
    Parameters
    ----------
    texts: list, list of texts (remove None and empty string)
    Returns
    -------
    key_phrases: list, list of top key phrases that expain the article
    """
    print('generating vocabulary candidate...')
    vocabulary = [generate_candidate(unidecode(text), method=method) for text in texts]
    vocabulary = list(chain(*vocabulary))
    vocabulary = list(np.unique(vocabulary)) # unique vocab
    with open('BBC_tfidf_NER_word.pkl', 'wb') as f:
        pickle.dump(vocabulary, f)
    print('done!')



    max_vocab_len = max(map(lambda s: len(s.split(' ')), vocabulary))
    tfidf_model = TfidfVectorizer(vocabulary=vocabulary, lowercase=True,
                                  ngram_range=(1,max_vocab_len), stop_words=None,
                                  min_df=min_df, max_df=max_df)
    X = tfidf_model.fit_transform(texts)

    vocabulary_sort = [v[0] for v in sorted(tfidf_model.vocabulary_.items(),
                                            key=operator.itemgetter(1))]
    sorted_array = np.fliplr(np.argsort(X.toarray()))

    # return list of top candidate phrase
    key_phrases = list()
    for sorted_array_doc in sorted_array:
        key_phrase = [vocabulary_sort[e] for e in sorted_array_doc[0:num_key]]
        key_phrases.append(key_phrase)

    return key_phrases

def get_list_of_texts(path):
    #path = "/Users/rossabramson/Documents/Junior NYU/NLP/Project/Data_Sources/SemEval2010/test"
    list_of_documents = []
    list_of_titles=[]
    for filename in os.listdir(path):
        list_of_titles.append(str(filename))
        print("Reading: "+str(filename))
        data=''
        with open("bbc/" + filename,'r') as myfile:
            data += myfile.read().replace('\n', ' ')
        list_of_documents.append(data)


    return list_of_documents, list_of_titles
def quicklyGet(texts, method='word', min_df=5, max_df=0.8, num_key=int(sys.argv[1])):
    # BBC_tfidf_NER_word.pkl
    # BBC_tfidf_Stem_word.pkl
    with open('BBC_tfidf_NER_word.pkl', 'rb') as f:
        vocabulary = pickle.load(f)

    max_vocab_len = max(map(lambda s: len(s.split(' ')), vocabulary))
    tfidf_model = TfidfVectorizer(vocabulary=vocabulary, lowercase=True,
                                  ngram_range=(1, max_vocab_len), stop_words=None,
                                  min_df=min_df, max_df=max_df)
    X = tfidf_model.fit_transform(texts)

    vocabulary_sort = [v[0] for v in sorted(tfidf_model.vocabulary_.items(),
                                            key=operator.itemgetter(1))]
    sorted_array = np.fliplr(np.argsort(X.toarray()))

    # return list of top candidate phrase
    key_phrases = list()
    for sorted_array_doc in sorted_array:
        key_phrase = [vocabulary_sort[e] for e in sorted_array_doc[0:num_key]]
        key_phrases.append(key_phrase)

    return key_phrases
if __name__ == '__main__':

    list_of_documents, titles=get_list_of_texts("bbc/")
    key_phrases=quicklyGet(list_of_documents)
    key_phrases = keyphrase_extraction_tfidf(list_of_documents)
    with open ("tfidf-output-BBC--word-10.txt",'w') as fileName:
        for i, keys in enumerate(key_phrases):
            fileName.write("Article: "+titles[i]+"\n")
            fileName.write(str(','.join(keys))+'\n\n')
    print('\n---DONE--\n')
