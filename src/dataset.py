from glob import glob
import csv
import pandas as pd
import numpy as np

DTYPES={
        'offer_id':       np.int64,
        'offer_len':      np.int64,
        'token':         str,
        'loc':            np.int64,
        'pos':           str,
        'pos_left':      str,
        'pos_right':     str,
        'token_len':      np.int64,
        'all_upper':       bool,
        'token_len':      np.int64,
        'n_tokens':      np.int64,
        'real_label':    str,
    }

def load_training_data():
    offer_files = sorted(glob('data/offers-*.csv'))
    headers = None
    records = []
    for offer_file in offer_files:
        with open(offer_file, 'r', encoding='utf-8-sig') as r:
            reader = csv.reader(r)
            headers = next(reader)
            for l in reader:
                records.append(l)
    frame = pd.DataFrame(records, columns=headers).astype(DTYPES)
    frame['real_label'] = frame['real_label'].replace('',np.nan)
    return frame
