import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *

nltk.download('stopwords')


english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
all_stopwords = english_stopwords.union(corpus_stopwords)
def tokenize(text):
    tokens = [token.group() for token in RE_WORD.finditer(str(text).lower())]
    return list(filter(lambda x: x not in all_stopwords, tokens))


stemmer = PorterStemmer()
def stemmeing(tokens):
    tokens = [stemmer.stem(t) for t in tokens]
    return tokens


