from io import open
import unicodedata
import re

from data import Data

class DataPreprocess(object):
    def __init__(self, max_length=10):
        self.max_length = max_length
        self.eng_prefixes = ("i am ", "i m ",
                             "he is", "he s ",
                             "she is", "she s",
                             "you are", "you re ",
                             "we are", "we re ",
                             "they are", "they re ")

    def read_langs(self, lang1, lang2, reverse=False):
        print("Reading lines...")

        # Read the file and split into lines
        lines = open('./Datasets/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
                read().strip().split('\n')

        # Split every line into pairs and normalize
        pairs = [[self.normalize_string(s) for s in l.split('\t')] for l in lines]

        # Reverse pairs, make Data instances
        if reverse:
            pairs = [list(reversed(p)) for p in pairs]
            input_lang = Data(lang2)
            output_lang = Data(lang1)
        else:
            input_lang = Data(lang1)
            output_lang = Data(lang2)

        return input_lang, output_lang, pairs

    # Turn a Unicode string to plain ASCII, thanks to
    # http://stackoverflow.com/a/518232/2809427
    def unicode_to_ascii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    # Lowercase, trim, and remove non-letter characters
    def normalize_string(self, s):
        s = self.unicode_to_ascii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        return s

    def filter_pair(self, p):
        return len(p[0].split(' ')) < self.max_length and \
            len(p[1].split(' ')) < self.max_length and \
            p[1].startswith(self.eng_prefixes)


    def filter_pairs(self, pairs):
        return [pair for pair in pairs if self.filter_pair(pair)]

    def prepare_data(self, lang1, lang2, reverse=False):
        input_lang, output_lang, pairs = self.read_langs(lang1, lang2, reverse)
        print("Read %s sentence pairs" % len(pairs))
        pairs = self.filter_pairs(pairs)
        print("Trimmed to %s sentence pairs" % len(pairs))
        print("Counting words...")
        for pair in pairs:
            input_lang.add_sentence(pair[0])
            output_lang.add_sentence(pair[1])
        print("Counted words:")
        print(input_lang.name, input_lang.n_words)
        print(output_lang.name, output_lang.n_words)
        return input_lang, output_lang, pairs
