import pandas as pd
import numpy as np
import string
from sklearn.base import BaseEstimator, TransformerMixin

class IsPunctuation(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.punctuation = set(string.punctuation)
        
    def fit(self, X, y=None):
        return self

    def transform(self, X):        
        punct = [False] * len(X)
        for i, t in enumerate(X):
            punct[i] = t in self.punctuation
        return np.array(punct).reshape(-1,1)


class RelativeLocations(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        return self

    def transform(self, X):        
        return (X['token_position']/X['offer_length']).values.reshape(-1,1)


class Reshaper(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        return self

    def transform(self, X):        
        return X.values.reshape(-1,1)
