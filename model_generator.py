import pickle

from nltk.corpus import wordnet as wn
import gensim
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
import nltk

nltk.download('wordnet')

import tokenizer

with open('id_to_text.pkl', 'rb') as inp:
    invi = pickle.load(inp)
sentences_to_train = [list(tokenizer.tokenize(text)) for text in list(invi.values())[0:1000]]
model = Word2Vec(sentences=sentences_to_train, vector_size=100, window=5, min_count=1, workers=4)
model.save("word2vec.model")
start = 1000
for stop in range(start + 1000, 350001, 1000):
    model = Word2Vec.load("word2vec.model")
    sentences_to_train = [list(tokenizer.tokenize(text)) for text in list(invi.values())[start:stop]]
    model.train(sentences_to_train, total_examples=1, epochs=1)
    print(stop)
    model.save("word2vec.model")
    start = stop

# print(common_texts)

# synonyms = []
# for syn in wn.synsets("best"):
#     for i in syn.lemmas():
#         synonyms.append(i.name())
# print(set(synonyms))
# vector = model.wv["run"]  # get numpy vector of a word
# print(vector)
# model = Word2Vec.load("word2vec.model")
# sims = model.wv.most_similar("men", topn=10)  # get other similar words
# print(sims)
