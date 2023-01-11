import nltk
import spark as spark
from gensim.models import Word2Vec
from google.cloud import storage

nltk.download('wordnet')

import tokenizer

import hashlib
def _hash(s):
    return hashlib.blake2b(bytes(s, encoding='utf8'), digest_size=5).hexdigest()
from pyspark.sql import *
from pyspark.sql.functions import *
from graphframes import *

bucket_name = 'id_to_title'
full_path = f"gs://{bucket_name}/"
paths=[]

client = storage.Client()
blobs = client.list_blobs(bucket_name)
for b in blobs:
    if b.name != 'graphframes.sh':
        paths.append(full_path+b.name)

parquetFile = spark.read.parquet(*paths)
doc_text_pairs = parquetFile.select("id", "text").rdd

invi = doc_text_pairs.collectAsMap()
sentences_to_train = [list(tokenizer.tokenize(text)) for text in list(invi.values())[0:1000]]
model = Word2Vec(sentences=sentences_to_train, vector_size=100, window=5, min_count=1, workers=4)
model.save("word2vec.model")
start = 1000
for stop in range(start + 1000, 50000, 1000):
    model = Word2Vec.load("word2vec.model")
    sentences_to_train = [list(tokenizer.tokenize(text)) for text in list(invi.values())[start:stop]]
    model.train(sentences_to_train, total_examples=1, epochs=1)
    print("word2vec working on: " + stop)
    model.save("word2vec.model")
    start = stop