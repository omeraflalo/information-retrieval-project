import nltk
# nltk.download('stopwords')
from nltk.stem.porter import *
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer



def tokenize(text):
    RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){,24}""", re.UNICODE)
    return [token.group() for token in RE_WORD.finditer(text.lower())]


print(tokenize("omer aflalo and pich 5 . , puts af put"))


# Getting tokens from the text while removing punctuations.
def filter_tokens(tokens, tokens2remove=None):
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
    for token in tokens:
        if token in tokens2remove:
            tokens = list(filter(lambda a: a != token, tokens))
    tokens = [stemmer.stem(t) for t in tokens]
    return set(tokens)

def tokenize_text(txt):
    return filter_tokens(tokenize(txt),
                    stopwords.words('english'))
print(tokenize_text("omer aflalo play was played bring brought and an pich 5 puts teach taught af put"))
