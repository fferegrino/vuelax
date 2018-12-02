def get_features(doc):
    return [token_to_features(doc, i) for i in range(len(doc))]


def get_labels(doc):
    return [doc[i].label for i in range(len(doc))]


def get_tokens(doc):
    return [doc[i].token for i in range(len(doc))]


def token_to_features(doc, i):
    tkn = doc[i]
    word = doc[i].token
    postag = doc[i].POS

    # Common features for all words. You may add more features here based on your custom use case
    features = [
        'bias',
        'word.lower=' + word.lower(),
        # 'word[-3:]=' + word[-3:],
        # 'word[-2:]=' + word[-2:],
        # 'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'word.isdigit=%s' % tkn.is_numeric,
        # 'word.ispunct=%s' % tkn.is_punctuation,
        # 'word.location=%s' % doc[i]['loc'],
        'postag=' + postag
    ]

    # Features for words that are not at the beginning of a document
    if i > 0:
        tkn1 = doc[i - 1]
        word1 = doc[i - 1].token
        postag1 = doc[i - 1].POS
        features.extend([
            '-1:word.lower=' + word1.lower(),
            '-1:word.istitle=%s' % word1.istitle(),
            # '-1:word.isupper=%s' % word1.isupper(),
            '-1:word.isdigit=%s' % tkn1.is_numeric,
            '-1:word.ispunct=%s' % tkn1.is_punctuation,
            '-1:postag=' + postag1
        ])
    else:
        # Indicate that it is the 'beginning of a document'
        features.append('BOS')

    # Features for words that are not at the end of a document
    if i < len(doc) - 1:
        tkn1 = doc[i + 1]
        word1 = doc[i + 1].token
        postag1 = doc[i + 1].POS
        features.extend([
            '+1:word.lower=' + word1.lower(),
            '+1:word.istitle=%s' % word1.istitle(),
            # '+1:word.isupper=%s' % word1.isupper(),
            '+1:word.isdigit=%s' % tkn1.is_numeric,
            '+1:word.ispunct=%s' % tkn1.is_punctuation,
            '+1:postag=' + postag1
        ])
    else:
        # Indicate that it is the 'end of a document'
        features.append('EOS')

    return features
