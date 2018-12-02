import pandas as pd
import pycrfsuite
from m16_mlutils.datatools.evaluation import eval_summary
from sklearn.model_selection import train_test_split

from extractor import get_labels, get_features
from features import row_to_tokenfeatures
import json
from dataset import load_training_data


def load_data():
    training_set = load_training_data()

    training_set = training_set[~pd.isna(training_set['real_label'])]

    return training_set


def train_eval_algorithm(dump_to_disk=False):
    training_set = load_data()

    documents = []
    current_doc = []
    prev = -1
    for i, word in training_set.iterrows():
        if i != prev:
            if current_doc:
                documents.append(current_doc)
            current_doc = []
        current_doc.append(row_to_tokenfeatures(word))
        prev = i

    if current_doc:
        documents.append(current_doc)

    train_docs, test_docs = train_test_split(documents)

    y_train = [get_labels(s) for s in train_docs]
    X_train = [get_features(s) for s in train_docs]

    y_test = [get_labels(s) for s in test_docs]
    X_test = [get_features(s) for s in test_docs]

    y_test_flat = [item for sublist in y_test for item in sublist]
    y_pred_flat = []

    trainer = pycrfsuite.Trainer(verbose=False)

    for xseq, yseq in zip(X_train, y_train):
        trainer.append(xseq, yseq)

    trainer.set_params({
        'c1': 1.0,  # coefficient for L1 penalty
        'c2': 1e-3,  # coefficient for L2 penalty
        'max_iterations': 50,  # stop earlier

        # include transitions that are possible, but not observed
        'feature.possible_transitions': True
    })

    trainer.train('models/vuelax.crf')

    crf_tagger = pycrfsuite.Tagger()
    crf_tagger.open('models/vuelax.crf')

    for doc in X_test:
        predicted = crf_tagger.tag(doc)
        y_pred_flat.extend(predicted)

    metrics, summary, cm = eval_summary(y_test_flat, y_pred_flat)

    return metrics, summary, cm


if __name__ == '__main__':
    metrics, summary, cm = train_eval_algorithm(True)

    with open('metrics.json', 'w') as w:
        json.dump(metrics.to_dict(), w, indent=4)
    with open('summary.txt', 'w') as w:
        w.write(summary)
