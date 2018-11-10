#!/usr/bin/env bash
download_tagger ()
{
    if [ ! -d tagger ]
    then
        mkdir tagger
        wget -O tagger/spanish.tagger https://github.com/fferegrino/vuelax/releases/download/0.1.0-circleci/spanish-distsim.tagger
        wget -O tagger/stanford-postagger.jar https://github.com/fferegrino/vuelax/releases/download/0.1.0-circleci/stanford-postagger.jar
    fi
}

download_data ()
{
    if [ ! -d data ]
    then
        mkdir data
        pipenv run kaggle datasets download -p data -d ioexception/vuelax
        unzip data/vuelax.zip -d data
        chmod 666 data/i__training_data.csv
        chmod 666 data/vuelos.csv
    fi
}

setup ()
{
    download_data
    download_tagger

    if [ ! -d models ]
    then
        mkdir models
    fi
}

"$@"
