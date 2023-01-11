import math

import numpy as np
from numpy.linalg import norm


def index_tfidf(tokenized_query, index, path):
    epsilon = .0000001
    d = {}
    for token in tokenized_query:
        for doc_id, tf in index.read_posting_list(path, token):
            if doc_id not in d.keys():
                d[doc_id] = dict(zip(tokenized_query, [0] * len(tokenized_query)))
            d[doc_id][token] = (tf / index.DL.get(doc_id)) * math.log(len(index.DL) / (index.df[token] + epsilon), 10)
    return d


def query_tfidf(tokenized_query, index):
    term_tf = {}
    for token in tokenized_query:
        term_tf[token] = term_tf.get(token, 0) + 1
    tfidf = {}
    for token, tf in term_tf.items():
        tfidf[token] = (tf / len(tokenized_query)) * math.log(len(index.DL) / (index.df[token]), 10)
    return tfidf


def temp(query, index, path):
    q_tfidf = np.array(list(query_tfidf(query, index).values()))
    sim = {}
    for doc_id, term_tfidf in index_tfidf(query, index, path).items():
        i_tfidf = np.array(list(term_tfidf.values()))
        sim[doc_id] = np.dot(q_tfidf, i_tfidf) / (norm(q_tfidf) * norm(i_tfidf))
    return sim
