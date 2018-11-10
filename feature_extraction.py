import pandas as pd
import numpy as np
from m16_mlutils.pipeline import CategoryEncoder, DataFrameSelector
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.externals import joblib
from sklearn.base import BaseEstimator, TransformerMixin

import os

HOME = os.getenv('HOME')

from pipelines import IsPunctuation, RelativeLocations, Reshaper



## Token Part-Of-the-Speech

pipeline_token_pos = Pipeline([
    ('selector', DataFrameSelector(['pos'])),
    ('encoder', CategoryEncoder())
])


## Is punctuation (more precise)

pipeline_is_punctuation = Pipeline([
    ('selector', DataFrameSelector(['token'])),
    ('is_punct', IsPunctuation())
])


## Relative location

pipeline_relative_location = Pipeline([
    ('location', RelativeLocations())
])


# ## Token length

pipeline_token_length = Pipeline([
    ('selector', DataFrameSelector(['offer_len', 'token_len'])),
    ('scaler', MaxAbsScaler())
])


# ## Unmodified features

pipeline_unmodified = Pipeline([
    ('select', DataFrameSelector('all_upper')),
    ('reshape', Reshaper())
])


# # Joint pipelines

joint_pipeline = Pipeline([
    ('get_features', FeatureUnion([
        ('token_pos', pipeline_token_pos),
        ('punctuation', pipeline_is_punctuation),
        ('rel_loc', pipeline_relative_location),
        ('token_length', pipeline_token_length),
        ('original_features', pipeline_unmodified)
    ]))
])

joblib.dump(joint_pipeline, 'models/features_pipeline.joblib')
