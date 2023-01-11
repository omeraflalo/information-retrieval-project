# import requests
# rescheck=requests.post('http://34.72.114.12:8080/get_pageview', json=[6941240,8554885,20611456])
# # i=rescheck.json()
# # t=3
# print((rescheck.json()))
# # 6941240   8554885  20611456

from sklearn.feature_extraction.text import TfidfVectorizer

import tokenizer

# list of documents
documents = ["This is the first document.", "This document is the second document.", "And this is the third one.", "Is this the first document?"]

# create the TfidfVectorizer object
vectorizer = TfidfVectorizer()



# fit and transform the documents
tfidf_matrix = vectorizer.fit_transform(documents)

# get the feature names
feature_names = vectorizer.get_feature_names()

# get the query
query = "Is this the first document?"

query = tokenizer.tokenize(query)

# find the index of the query in the feature names
query_index = feature_names.index(query)

# get the tf-idf score of the query
query_tfidf = tfidf_matrix[query_index]

print(query_tfidf)
