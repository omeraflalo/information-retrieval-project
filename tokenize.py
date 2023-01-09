import re
import nltk
from nltk.stem.porter import *


def tokenize(text):
    RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){,24}""", re.UNICODE)
    return [token.group() for token in RE_WORD.finditer(text.lower())]
print(tokenize("omer aflalo and pich 5 puts af put"))


# Getting tokens from the text while removing punctuations.
def filter_tokens(tokens, tokens2remove=None, use_stemming=False):
  ''' The function takes a list of tokens, filters out `tokens2remove` and
      stem the tokens using `stemmer`.
  Parameters:
  -----------
  tokens: list of str.
    Input tokens.
  tokens2remove: frozenset.
    Tokens to remove (before stemming).
  use_stemming: bool.
    If true, apply stemmer.stem on tokens.
  Returns:
  --------
  list of tokens from the text.
  '''
  stemmer = PorterStemmer()
  if tokens2remove is None:
    return tokens
  elif use_stemming:
    tokens = [stemmer.stem(t)  for t in tokens]
  else:
    for t in tokens:
      if t in tokens2remove:
        tokens = list(filter(lambda a: a!=t, tokens))
  return tokens

print(filter_tokens(tokenize("omer aflalo and pich 5 puts af put"),["a"],True))