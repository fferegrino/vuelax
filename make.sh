#!/usr/bin/env bash

IN_PIPENV="pipenv run"

download_tagger ()
{
    if [ ! -d bin ]
    then
        mkdir bin
        wget -O bin/spanish.tagger https://github.com/fferegrino/vuelax/releases/download/0.1.0-circleci/spanish-distsim.tagger
        wget -O bin/stanford-postagger.jar https://github.com/fferegrino/vuelax/releases/download/0.1.0-circleci/stanford-postagger.jar
    fi
}

setup ()
{
    download_tagger

    if [ ! -d bin ]
    then
        mkdir bin
    fi
}

prepare_dvc ()
{
    $IN_PIPENV invoke merge-data --path data/raw --file vuelax.csv
}

upload_kaggle ()
{
    mkdir data/interim/kaggle
    $IN_PIPENV kaggle datasets metadata -p ./ ioexception/vuelax
    mv dataset-metadata.json data/interim/kaggle
    cp data/raw/vuelos.csv data/interim/kaggle
    export EXPERIMENT_ROOT=$PWD
    export PYTHONPATH=$PWD/src
    $IN_PIPENV invoke merge-data --path data/interim/kaggle
    $IN_PIPENV invoke prepare-kaggle-metadata --meta-file data/interim/kaggle/dataset-metadata.json
    $IN_PIPENV kaggle datasets version -p data/interim/kaggle -m "Updated data"
    rm -rf data/interim/kaggle
}

start ()
{
    export EXPERIMENT_ROOT=$PWD
    export PYTHONPATH=$PWD/src
    $IN_PIPENV jupyter notebook
}

"$@"
