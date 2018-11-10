from classify import train_eval_algorithm
import json


def test_metrics():
    with open('metrics.json', 'r') as r:
        metrics = json.load(r)
    assert metrics['precision'] > 0.9
    assert metrics['recall'] > 0.8
