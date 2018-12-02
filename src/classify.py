import pandas as pd
from m16_mlutils.datatools.evaluation import eval_summary
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from pipelines import create_pipeline
import json
from dataset import load_training_data


def load_data():
    training_set = load_training_data()

    training_set = training_set[~pd.isna(training_set['real_label'])]

    return training_set


def train_eval_algorithm(dump_to_disk=False):
    training_set = load_data()
    pipeline = create_pipeline(True)

    best_classifier = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                                             max_depth=None, max_features='auto', max_leaf_nodes=None,
                                             min_impurity_decrease=0.0, min_impurity_split=None,
                                             min_samples_leaf=1, min_samples_split=2,
                                             min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=None,
                                             oob_score=False, random_state=42, verbose=0, warm_start=False)
    X_train, X_test, y_train, y_test = train_test_split(training_set, training_set['real_label'])
    pipeline.steps.append(('clf', best_classifier))

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    metrics, summary, cm = eval_summary(y_test, y_pred)

    if dump_to_disk:
        joblib.dump(pipeline, 'models/classify_pipeline.joblib')

    return metrics, summary, cm


if __name__ == '__main__':
    metrics, summary, cm = train_eval_algorithm(True)

    with open('metrics.json', 'w') as w:
        json.dump(metrics.to_dict(), w, indent=4)
    with open('summary.txt', 'w') as w:
        w.write(summary)
