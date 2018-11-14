import string

import numpy as np
import pandas as pd
from m16_mlutils.pipeline import CategoryEncoder, DataFrameSelector
from nltk.tag.stanford import StanfordPOSTagger
from nltk.tokenize import TweetTokenizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import MaxAbsScaler


class SentenceChunker(BaseEstimator, TransformerMixin):
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
        offer_len = [len(label) for l in only_tokens]
        n_tokens = [len(only_tokens) for l in only_tokens]
        augmented = ['<p>'] + tags + ['</p>']
        uppercase = [all([l.isupper() for l in token]) for token in only_tokens]
        return offer_len, only_tokens, positions, tags, augmented[:len(only_tokens)], augmented[
                                                                                      2:], lengths, uppercase, n_tokens

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        dictionaries = []
        for sentence in X:
            tags = self.__process_label(sentence)
            dictionary = {
                'offer_len': tags[0],
                'token': tags[1],
                'loc': tags[2],
                'pos': tags[3],
                'pos_left': tags[4],
                'pos_right': tags[5],
                'token_len': tags[6],
                'all_upper': tags[7],
                'n_tokens': tags[8]
            }
            dictionaries.append(pd.DataFrame(dictionary))
        df = pd.concat(dictionaries)
        return df


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


def create_pipeline(dump_to_disk=False):
    pipeline_token_pos = Pipeline([
        ('selector', DataFrameSelector(['pos'])),
        ('encoder', CategoryEncoder())
    ])

    pipeline_is_punctuation = Pipeline([
        ('selector', DataFrameSelector(['token'])),
        ('is_punct', IsPunctuation())
    ])

    pipeline_relative_location = Pipeline([
        ('location', RelativeLocations())
    ])

    pipeline_token_length = Pipeline([
        ('selector', DataFrameSelector(['offer_len', 'token_len'])),
        ('scaler', MaxAbsScaler())
    ])

    pipeline_unmodified = Pipeline([
        ('select', DataFrameSelector('all_upper')),
        ('reshape', Reshaper())
    ])

    joint_pipeline = Pipeline([
        ('get_features', FeatureUnion([
            ('token_pos', pipeline_token_pos),
            ('punctuation', pipeline_is_punctuation),
            ('rel_loc', pipeline_relative_location),
            ('token_length', pipeline_token_length),
            ('original_features', pipeline_unmodified)
        ]))
    ])

    if dump_to_disk:
        joblib.dump(joint_pipeline, 'models/features_pipeline.joblib')

    return joint_pipeline
