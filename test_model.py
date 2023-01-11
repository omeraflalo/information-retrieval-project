from gensim.models import Word2Vec


def print_samiliar(model, word):
    sims = model.wv.most_similar(word, topn=5)  # get other similar words
    print(word + ": " + str(sims))


model = Word2Vec.load("word2vec111_chank_49000.model")

print_samiliar(model, "marvel")
print_samiliar(model, "man")
print_samiliar(model, "superman")
print_samiliar(model, "friend")
print_samiliar(model, "best")
print_samiliar(model, "kid")
print_samiliar(model, "bird")
print_samiliar(model, "chess")
# print_samiliar(model, "word2vec")
print_samiliar(model, "computer")
print_samiliar(model, "run")
print_samiliar(model, "rain")
print_samiliar(model, "sex")
print_samiliar(model, "floor")
print_samiliar(model, "israel")
print_samiliar(model, "files")
print_samiliar(model, "games")
print_samiliar(model, "england")
