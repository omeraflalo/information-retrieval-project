import math

import numpy as np


def index_tfidf(tokenized_query, index, path):
    epsilon = .0000001
    d = {}
    for token in tokenized_query:
        for doc_id, tf in index.read_posting_list(path, token):
            if not d.get(doc_id):
                d[doc_id] = dict(zip(tokenized_query, [0] * len(tokenized_query)))
            d[doc_id][token] = (tf / index.DL.get(doc_id)) * math.log(len(index.DL) / (index.df[token] + epsilon), 10)
    return d


def query_tfidf(tokenized_query, index):
    term_tf = {}
    for token in tokenized_query:
        term_tf[token] = term_tf.get(token, 0) + 1
    tfidf = {}
    tfi = np.array()
    for token, tf in term_tf.items():
        tfidf[token] = (tf / len(tokenized_query)) * math.log(len(index.DL) / (index.df[token]), 10)
    return tfidf
