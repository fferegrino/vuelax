from nltk.tokenize import TweetTokenizer
from collections import namedtuple
from nltk.tag.stanford import StanfordPOSTagger
import string

tokenizer = TweetTokenizer()


def index_emoji_tokenize(sentence, return_flags=False):
    flag = ''
    ix = 0
    for t in tokenizer.tokenize(sentence):
        ix = sentence.find(t, ix)
        if len(t) == 1 and ord(t) >= 127462:  # this is the code for 🇦
            if not return_flags: continue
            if flag:
                yield flag + t, ix - 1
                flag = ''
            else:
                flag = t
        else:
            yield t, ix
        ix = +1


punctuation = set(string.punctuation)


def is_punctuation(token):
    return token in punctuation


def is_numeric(token):
    try:
        float(token.replace(",", ""))
        return True
    except:
        return False


TokenFeatures = namedtuple('TokenFeatures',
                           ['sentence_id',
                            'offer_length',
                            'token',
                            'position',
                            'POS',
                            'left_POS',
                            'right_POS',
                            'token_length',
                            'uppercase',
                            'tokens_in_sentence',
                            'is_numeric',
                            'is_punctuation',
                            'label'
                            ])


def row_to_tokenfeatures(row):
    sentence_id = int(row.name)
    word_dictionary = row.to_dict()
    word_dictionary['sentence_id'] = sentence_id
    return TokenFeatures(sentence_id,
                         word_dictionary['offer_len'],
                         word_dictionary['token'],
                         word_dictionary['loc'],
                         word_dictionary['pos'],
                         word_dictionary['pos_left'],
                         word_dictionary['pos_right'],
                         word_dictionary['token_len'],
                         word_dictionary['all_upper'],
                         word_dictionary['n_tokens'],
                         is_numeric(word_dictionary['token']),
                         is_punctuation(word_dictionary['token']),
                         word_dictionary['real_label'])


class SentenceProcessor:

    def __init__(self, tagger, jar):
        self.__tagger = StanfordPOSTagger(tagger, jar)

    def process(self, sentence, sentence_id=None):
        tokens = list(index_emoji_tokenize(sentence, True))
        only_tokens = [l[0] for l in tokens]
        positions = [l[1] for l in tokens]
        tagged = self.__tagger.tag(only_tokens)
        tags = [l[1] for l in tagged]
        lengths = [len(l) for l in only_tokens]
        offer_len = [len(sentence) for l in only_tokens]
        n_tokens = [len(only_tokens) for l in only_tokens]
        augmented = ['<p>'] + tags + ['</p>']
        uppercase = [all([l.isupper() for l in token]) for token in only_tokens]
        numeric = [is_numeric(l) for l in only_tokens]
        puntctuations = [is_punctuation(l) for l in only_tokens]
        labels = [None for l in only_tokens]
        sentence_ids = [sentence_id] * len(only_tokens)
        return [TokenFeatures(*t) for t in
                zip(sentence_ids, offer_len, only_tokens, positions, tags, augmented[:len(only_tokens)], augmented[2:],
                    lengths, uppercase, n_tokens, numeric, puntctuations, labels)]
