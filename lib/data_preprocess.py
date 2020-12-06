import re
import torch
from torch.utils.data import Dataset
from .data_handle import load_data


class TrainSet(Dataset):

    def __init__(self, ko_path: str, en_path: str, ko_vocab, en_vocab):
        self._ko_vocab = ko_vocab
        self._en_vocab = en_vocab

        self._ko_corpus = _preprocessor(load_data(ko_path))
        self._en_corpus = _preprocessor(load_data(en_path))
        self.ko_w2idx = self._ko_vocab.load(self._ko_corpus)
        self.en_w2idx = self._en_vocab.load(self._en_corpus)
        self._data = self.dataset_form()

    def dataset_form(self):
        rst = []
        for ko, en in zip(self._ko_corpus, self._en_corpus):
            ko = [self._ko_vocab[x] for x in ko]
            en = [self._en_vocab[x] for x in en]
            # padding
            rst.append([ko, en])
        return rst

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
        print(idx, len(self.word2idx))
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


def _preprocessor(corpus: list):
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
