#!/usr/bin/env bash
if [ ! -d tagger ]
then
	mkdir tagger
	wget -O tagger/spanish.tagger https://github.com/fferegrino/vuelax/releases/download/0.1.0-circleci/spanish-distsim.tagger
	wget -O tagger/stanford-postagger.jar https://github.com/fferegrino/vuelax/releases/download/0.1.0-circleci/stanford-postagger.jar
fi

if [ ! -d data ]
then
    mkdir data
    pipenv run kaggle datasets download -p data -d ioexception/vuelax
    unzip data/vuelax.zip -d data
fi