import os
import pickle

import nltk
from gensim.models import Word2Vec

import tokenizer

nltk.download('wordnet')


def iterate(last, doc_num, start=0):
    for stop in range(start, len(invi), 1000):
        sentences_to_train = [list(tokenizer.tokenize(text)) for text in list(invi)[start:stop]]
        model.train(sentences_to_train, total_examples=1, epochs=1)
        part = str(doc_num) + "_chank_" + str(stop)
        new_file_name = "word2vec" + part + ".model"
        print("word2vec working on file " + part)
        model.save(new_file_name)
        os.remove(last)
        last = new_file_name
        start = stop
    return last


with open(f'texts/corpus_to_train_0.pkl', 'rb') as inp:
    invi = pickle.load(inp)
sentences_to_train = [list(tokenizer.tokenize(text)) for text in list(invi)[0:1000]]
model = Word2Vec(sentences=sentences_to_train, vector_size=100, window=5, min_count=1, workers=4)
model.save("word2vec.model")
last = "word2vec.model"
last = iterate(last, 0, 1000)
for i in range(1, 112):
    with open(f'texts/corpus_to_train_{i}.pkl', 'rb') as inp:
        invi = pickle.load(inp)
    last = iterate(last, i)

# model = Word2Vec.load("word2vec.model")
# sims = model.wv.most_similar("marvel", topn=10)  # get other similar words
# print(sims)
