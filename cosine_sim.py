import numpy as np


def cosine_similarity(D, Q):
    """
    Calculate the cosine similarity for each candidate document in D and a given query (e.g., Q).
    Generate a dictionary of cosine similarity scores
    key: doc_id
    value: cosine similarity score

    Parameters:
    -----------
    D: DataFrame of tfidf scores.

    Q: vectorized query with tfidf scores

    Returns:
    -----------
    dictionary of cosine similarity score as follows:
                                                                key: document id (e.g., doc_id)
                                                                value: cosine similarty score.
    """
    cosine_similarity_dic = {}
    for x in D.index.values:
        cosine_similarity_dic[x] = (np.dot(D.loc[[x]], Q) / (np.linalg.norm(D.loc[[x]]) * np.linalg.norm(Q)))[0]
    return cosine_similarity_dic
