import re
import torch
from torch.utils.data import Dataset
from .data_handle import load_data


class Corpus(Dataset):

    def __init__(self, data_set):
        self._data = data_set

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx: int):
        """ return input word and target word """
        ko, en = self._data[idx]
        return torch.tensor(ko), torch.tensor(en)


UNKNOWN = '[UKN]'
CLOSE = '[CLS]'
PAD = '[PAD]'
START = '[SRT]'


class Vocab:
    # 0 for padding
    WORD2IDX = {PAD: 0, UNKNOWN: 1, CLOSE: 2, START: 3}

    def __init__(self, min_cnt):
        self.min_cnt = min_cnt
        self.excepts = '.#$%^&*'
        self.word2idx = {k: v for k, v in self.WORD2IDX.items()}
        self.idx2word = {}

    def load(self, corpus: list):
        vocabs = {}
        for sent in corpus:
            for word in sent[1: -1]:
                if word not in vocabs.keys():
                    vocabs[word] = 0
                vocabs[word] += 1
        idx = len(self.WORD2IDX)
        for w in vocabs.keys():
            if self._vocabs_filter(w, vocabs[w]):
                self.word2idx[w] = idx
                idx += 1
        # print(idx, len(self.word2idx))
        return self.word2idx

    def _vocabs_filter(self, v, cnt):
        if cnt < self.min_cnt:
            return
        if v in self.excepts:
            return
        return v

    def to_idx2word(self):
        self.idx2word = {idx: w for w, idx in self.word2idx.items()}

    def get_word(self, idx):
        return self.idx2word[idx]

    def __len__(self):
        return len(self.word2idx)

    def __getitem__(self, item):
        try:
            return self.word2idx[item]
        except KeyError:
            return self.word2idx[UNKNOWN]


def preprocessor(corpus: list):
    result = []
    for line in corpus:
        sents = line.strip()
        words = _to_word(sents)
        if len(words) < 5:
            continue
        words = [START] + words + [CLOSE]
        result.append(words)
    return result


def _to_word(a_sent: str) -> list:
    """ Filter word as stop words """
    rst = []
    for word in re.split(r'(\s|\.|\,|\?|\!|\"|\')+', a_sent):
        word = word.strip()
        if not word or word in ',."\'':
            continue
        rst.append(word)
    return rst


def collate_fn(batch):
    ko, en = zip(*batch)
    pad_ko = torch.nn.utils.rnn.pad_sequence(ko,batch_first=True)
    pad_en = torch.nn.utils.rnn.pad_sequence(en,batch_first=True)
    return pad_ko, pad_en
