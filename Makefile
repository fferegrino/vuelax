download_tagger:
	mkdir tagger
	wget -O tagger/spanish.tagger https://github.com/fferegrino/vuelax/releases/download/0.1.0-circleci/spanish-distsim.tagger
	wget -O tagger/stanford-postagger.jar https://github.com/fferegrino/vuelax/releases/download/0.1.0-circleci/stanford-postagger.jar
