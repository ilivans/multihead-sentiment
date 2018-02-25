import numpy as np


class WordsEncoder():

    def __init__(self, max_len=None):
        self.word_to_id = dict()
        self.max_len = max_len
        self.vocabulary = []
        self.voc_size = 0

    def _words_to_ids(self, seq):
        ids = [self.word_to_id[w] for w in seq if w in self.word_to_id]
        ids = ids[:self.max_len - 1]
        ids += [0] * (self.max_len - len(ids))
        return ids

    def fit(self, X, y=None, len_percentile=99):
        if self.max_len is None:
            self.max_len = int(np.percentile(list(map(len, X)), len_percentile)) + 1
        self.vocabulary = list(set([word for x in X for word in x]))
        self.vocabulary = ['<EOS>'] + self.vocabulary
        self.voc_size = len(self.vocabulary)
        self.word_to_id = dict(zip(self.vocabulary, range(len(self.vocabulary))))

    def transform(self, X, y=None):
        return np.array(list(map(self._words_to_ids, X)))

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)
