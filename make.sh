#!/usr/bin/env bash

IN_PIPENV="pipenv run"

download_tagger ()
{
    if [ ! -d tagger ]
    then
        mkdir tagger
        wget -O tagger/spanish.tagger https://github.com/fferegrino/vuelax/releases/download/0.1.0-circleci/spanish-distsim.tagger
        wget -O tagger/stanford-postagger.jar https://github.com/fferegrino/vuelax/releases/download/0.1.0-circleci/stanford-postagger.jar
    fi
}

setup ()
{
    download_tagger

    if [ ! -d models ]
    then
        mkdir models
    fi
}

upload_kaggle ()
{
    mkdir data/kaggle
    kaggle datasets metadata -p ./ ioexception/vuelax
    mv dataset-metadata.json data/kaggle
    cp data/vuelos.csv data/kaggle
    export PYTHONPATH=src
    pipenv run invoke prepare-kaggle-data --path data/kaggle
    pipenv run invoke prepare-kaggle-metadata --meta-file data/kaggle/dataset-metadata.json
    kaggle datasets version -p data/kaggle -m "Updated data"
    rm -rf data/kaggle
}

start ()
{
    export PYTHONPATH=src
    $IN_PIPENV jupyter notebook
}

"$@"
