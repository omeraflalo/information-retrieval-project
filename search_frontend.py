import pickle
from collections import Counter

from flask import Flask, request, jsonify
from gensim.models import Word2Vec

import config
import cosine_sim
import tokenizer
import top_files
import math
from inverted_index_gcp import InvertedIndex


class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

body_index = InvertedIndex().read_index(config.path_to_body_index, "index")
title_index = InvertedIndex().read_index(config.path_to_title_index, "index")
anchor_index = InvertedIndex().read_index(config.path_to_anchor_index, "index")
# body_stemming_index = InvertedIndex().read_index(config.path_to_body_stemming_index, "index")

with open("pkl/id_to_title.pkl", 'rb') as f:
    id_to_title = pickle.load(f)

with open("pkl/page_rank.pkl", 'rb') as f:
    page_rank = pickle.load(f)

with open("pkl/page_views.pkl", 'rb') as f:
    page_views = pickle.load(f)

with open("pkl/DL_body.pkl", 'rb') as f:
    body_index.DL = pickle.load(f)

with open("pkl/DL_title.pkl", 'rb') as f:
    title_index.DL = pickle.load(f)

word2vec = Word2Vec.load("word2vec111_chank_49000.model")


def title_from_id_list(lst):
    return list(map(lambda x: (x[0], id_to_title[int(x[0])]), lst))


def calculate_cosin_sim(query, index_to_sim, path, stemming=True):
    tokenized_query = tokenizer.tokenize(query)
    if stemming:
        tokenized_query = tokenizer.stemmeing(tokenized_query)
    return cosine_sim.temp(tokenized_query, index_to_sim, path)


def matching_terms(tokenized_query, index_to_match, path):
    dic = {}
    for token in set(tokenized_query):
        if index_to_match.df.get(token):
            for doc, tf in index_to_match.read_posting_list(path, token):
                dic[doc] = dic.get(doc, 0) + 1
    return dic


@app.route("/search")
def search():
    """ Returns up to a 100 of your best search results for the query. This is
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    """
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    tokenized_query = tokenizer.tokenize(query)
    # add = []
    # for token in tokenized_query:
    #     if token in word2vec.wv.key_to_index:
    #         add += list(map(lambda x: x[0], filter(lambda x: x[1] > 0.85, word2vec.wv.most_similar(token, topn=5))))
    # print(add)
    # tokenized_query += add
    body_result = Counter(matching_terms(tokenized_query, body_index, config.path_to_body_index)).most_common()
    title_result = matching_terms(tokenized_query, title_index, config.path_to_title_index)
    # anchor_result = matching_terms(tokenized_query, anchor_index, config.path_to_anchor_index)
    counter = 0

    for doc_id, score in body_result:
        title_score = title_result.get(doc_id)
        # anchor_score = anchor_result.get(doc_id)
        if title_score:  # and anchor_score:
            counter += 1
            res.append(
                (doc_id, score + title_score, page_rank.get(doc_id, 0), page_views.get(doc_id, 0)))
        if counter == 300:
            break

    print(len(res))
    res = sorted(res, key=lambda x: (-x[1], -x[3], -x[2]))[:100]

    res = title_from_id_list(res)
    # print(len(res))

    return jsonify(res)


@app.route("/search_body")
def search_body():
    """ Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the
        staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    """
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    body_cosine = calculate_cosin_sim(query, body_index, config.path_to_body_index, False)
    res = top_files.get_top_n(body_cosine, id_to_title, 100)

    # END SOLUTION
    return jsonify(res)


@app.route("/search_title")
def search_title():
    """ Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF
        DISTINCT QUERY WORDS that appear in the title. DO NOT use stemming. DO
        USE the staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords. For example, a document
        with a title that matches two distinct query words will be ranked before a
        document with a title that matches only one distinct query word,
        regardless of the number of times the term appeared in the title (or
        query).

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    """
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    tokenized_query = tokenizer.tokenize(query)
    res = Counter(matching_terms(tokenized_query, title_index, config.path_to_title_index)).most_common()
    res = title_from_id_list(res)

    return jsonify(res)


@app.route("/search_anchor")
def search_anchor():
    """ Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE ANCHOR TEXT of articles, ordered in descending order of the
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page.
        DO NOT use stemming. DO USE the staff-provided tokenizer from Assignment
        3 (GCP part) to do the tokenization and remove stopwords. For example,
        a document with a anchor text that matches two distinct query words will
        be ranked before a document with anchor text that matches only one
        distinct query word, regardless of the number of times the term appeared
        in the anchor text (or query).

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    """
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    tokenized_query = tokenizer.tokenize(query)
    res = Counter(matching_terms(tokenized_query, anchor_index, config.path_to_anchor_index)).most_common()
    res = title_from_id_list(res)

    return jsonify(res)


@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    """ Returns PageRank values for a list of provided wiki article IDs.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    """
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)

    for doc_id in wiki_ids:
        res.append(page_rank.get(doc_id, 0))

    return jsonify(res)


@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    """ Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the
          provided list article IDs.
    """
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)

    for doc_id in wiki_ids:
        res.append(page_views.get(doc_id, 0))

    return jsonify(res)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=False)
