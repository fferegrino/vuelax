import pandas as pd
import numpy as np
import string
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.tag.stanford import StanfordPOSTagger
from nltk.tokenize import TweetTokenizer


class SentenceChunker(BaseEstimator, TransformerMixin):
    """
chunker = SentenceChunker(HOME + '/stanford_nlp/stanford-postagger-full-2018-02-27/models/spanish-distsim.tagger',
                HOME + '/stanford_nlp/stanford-postagger-full-2018-02-27/stanford-postagger.jar')

chunker.fit(['¡CUN a Holanda $8,885! Sin escala EE.UU'])
chunker.transform(['¡CUN a Holanda $8,885! Sin escala EE.UU'])
    """
    __TOKENIZER = TweetTokenizer()

    @staticmethod
    def index_emoji_tokenize(sentence, return_flags=False):
        i = 0
        flag = ''
        ix = 0
        for t in SentenceChunker.__TOKENIZER.tokenize(sentence):
            ix = sentence.find(t, ix)
            if len(t) == 1 and ord(t) >= 127462:  # this is the code for 🇦
                if not return_flags: continue
                if flag:
                    yield flag + t, ix - 1
                    flag = ''
                else:
                    flag = t
            else:
                yield t, ix
            ix = +1

    def __init__(self, model, jar_file):
        self._tagger = StanfordPOSTagger(model, jar_file)

    def __process_label(self, label, debug=False):
        tokens = list(SentenceChunker.index_emoji_tokenize(label, True))
        only_tokens = [l[0] for l in tokens]
        positions = [l[1] for l in tokens]
        tagged = self._tagger.tag(only_tokens)
        tags = [l[1] for l in tagged]
        lengths = [len(l) for l in only_tokens]
        n_tokens = [len(only_tokens) for l in only_tokens]
        augmented = ['<p>'] + tags + ['</p>']
        uppercase = [all([l.isupper() for l in token]) for token in only_tokens]
        return only_tokens, positions, tags, augmented[:len(only_tokens)], augmented[2:], lengths, uppercase, n_tokens

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        tags = []
        for sentence in X:
            tags.append(self.__process_label(sentence))
        return tags


class IsPunctuation(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.punctuation = set(string.punctuation)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        punct = [False] * len(X)
        for i, t in enumerate(X):
            punct[i] = t in self.punctuation
        return np.array(punct).reshape(-1, 1)


class RelativeLocations(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return (X['loc'] / X['offer_len']).values.reshape(-1, 1)


class Reshaper(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.values.reshape(-1, 1)
