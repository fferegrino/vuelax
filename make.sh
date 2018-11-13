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

setup ()
{
    download_tagger

    if [ ! -d models ]
    then
        mkdir models
    fi
}

"$@"
