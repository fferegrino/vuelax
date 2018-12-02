from glob import glob
import csv
from collections import namedtuple
import json
import pandas as pd
import numpy as np
from src.features import is_punctuation, is_numeric

DTYPES = {
    'offer_id': np.int64,
    'offer_len': np.int64,
    'token': str,
    'loc': np.int64,
    'pos': str,
    'pos_left': str,
    'pos_right': str,
    'token_len': np.int64,
    # 'all_upper':       bool,
    'token_len': np.int64,
    'n_tokens': np.int64,
    'real_label': str,
}

VUELOS_FIELDS = [
    {
        "name": "id",
        "title": "Just an identifier",
        "type": "Integer"
    },
    {
        "name": "label",
        "title": "The text for this offer",
        "type": "String"
    },
    {
        "name": "url",
        "title": "The original url of this offer",
        "type": "Uri"
    },
    {
        "name": "date",
        "title": None,
        "type": "DateTime"
    },
    {
        "name": "source",
        "title": None,
        "type": "String"
    }
]

TRAINING_DATA_FIELDS = [
    {
        "name": "offer_id",
        "title": None,
        "type": "Integer"
    },
    {
        "name": "offer_len",
        "title": None,
        "type": "Integer"
    },
    {
        "name": "token",
        "title": None,
        "type": "String"
    },
    {
        "name": "loc",
        "title": None,
        "type": "Integer"
    },
    {
        "name": "pos",
        "title": None,
        "type": "String"
    },
    {
        "name": "pos_left",
        "title": None,
        "type": "String"
    },
    {
        "name": "pos_right",
        "title": None,
        "type": "String"
    },
    {
        "name": "token_len",
        "title": None,
        "type": "Integer"
    },
    {
        "name": "all_upper",
        "title": None,
        "type": "Boolean"
    },
    {
        "name": "n_tokens",
        "title": None,
        "type": "Integer"
    },
    {
        "name": "real_label",
        "title": None,
        "type": "String"
    }
]


def __upper_boolean(value):
    if value.lower() == 'true':
        return True
    elif value.lower() == 'false':
        return False


def load_training_data(drop_no_label=True):
    offer_files = sorted(glob('data/offers-*.csv'))
    headers = None
    records = []
    for offer_file in offer_files:
        with open(offer_file, 'r', encoding='utf-8-sig') as r:
            reader = csv.reader(r)
            headers = next(reader)
            for l in reader:
                records.append(l)

    frame = pd.DataFrame(records, columns=headers).astype(DTYPES).set_index('offer_id')
    frame['all_upper'] = frame['all_upper'].apply(__upper_boolean)
    frame['real_label'] = frame['real_label'].replace('', np.nan)
    if drop_no_label:
        frame = frame.dropna(subset=['real_label']).copy()

    return frame


TokenFeatures = namedtuple('TokenFeatures',
                           ['sentence_id',
                            'offer_length',
                            'token',
                            'position',
                            'POS',
                            'left_POS',
                            'right_POS',
                            'token_length',
                            'uppercase',
                            'tokens_in_sentence',
                            'is_numeric',
                            'is_punctuation',
                            'label'
                            ])


def row_to_tokenfeatures(row):
    sentence_id = int(row.name)
    word_dictionary = row.to_dict()
    word_dictionary['sentence_id'] = sentence_id
    return TokenFeatures(sentence_id,
                         word_dictionary['offer_len'],
                         word_dictionary['token'],
                         word_dictionary['loc'],
                         word_dictionary['pos'],
                         word_dictionary['pos_left'],
                         word_dictionary['pos_right'],
                         word_dictionary['token_len'],
                         word_dictionary['all_upper'],
                         word_dictionary['n_tokens'],
                         is_numeric(word_dictionary['token']),
                         is_punctuation(word_dictionary['token']),
                         word_dictionary['real_label'])


def modify_metadata(metadata_file):
    with open(metadata_file, 'r') as r:
        meta = json.load(r)

    [i__training_data, vuelos] = meta['resources']

    i__training_data['path'] = i__training_data['path'].split('/')[-1]
    i__training_data['schema']['fields'] = TRAINING_DATA_FIELDS

    vuelos['path'] = vuelos['path'].split('/')[-1]
    vuelos['schema']['fields'] = VUELOS_FIELDS

    with open(metadata_file, 'w') as w:
        json.dump(meta, w, indent=4)
