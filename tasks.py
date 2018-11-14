from invoke import task

from src.dataset import load_training_data, modify_metadata
import os


@task
def prepare_kaggle_data(c, path='.', file='i__training_data.csv'):
    output_file = os.path.join(path, file)
    print(output_file)
    i__training_data = load_training_data()
    i__training_data.to_csv(output_file, encoding='utf-8-sig')


@task
def prepare_kaggle_metadata(c, meta_file):
    modify_metadata(meta_file)
