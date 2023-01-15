import math

import numpy as np
from numpy.linalg import norm

import tfidf


def temp(query, index, path):
    q_tfidf = np.array(list(tfidf.query_tfidf(query, index).values()))
    sim = {}
    norm_q_tfidf = norm(q_tfidf)
    for doc_id, term_tfidf in tfidf.index_tfidf(query, index, path).items():
        i_tfidf = np.array(list(term_tfidf.values()))
        sim[doc_id] = np.dot(q_tfidf, i_tfidf) / (norm_q_tfidf * norm(i_tfidf))
    return sim
