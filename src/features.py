
        
from nltk.tokenize import TweetTokenizer
import string

tokenizer = TweetTokenizer()

def index_emoji_tokenize(sentence, return_flags=False):
    i = 0
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