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
            ko = [self.ko_w2idx[x] for x in ko]
            en = [self.en_w2idx[x] for x in en]
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


class Vocab:
    # 0 for padding
    WORD2IDX = {UNKNOWN: 1, CLOSE: 2}

    def __init__(self, min_cnt):
        self.min_cnt = min_cnt
        self.excepts = '?!.#$%^&*'
        self.word2idx = self.WORD2IDX
        self.idx2word = {}

    def load(self, corpus: list):
        vocabs = {}
        for sent in corpus:
            for word in sent:
                if word not in vocabs.keys():
                    vocabs[word] = 0
                vocabs[word] += 1

        vocabs = vocabs.keys()
        self.word2idx = {w: idx for idx, w in enumerate(vocabs)}
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
    # 데이터 토크나이저 with options(최대 빈도 단어.. 등)
    #   데이터 문장 단위 분리
    #   데이터 단어 분리
    #   return 문장:[단어1, 단어2 ...]
    result = []
    for line in corpus:
        sents = line.strip()
        for s in sents:
            words = _to_word(s) + [CLOSE]
            result.append(words)
    return result


def _to_word(a_sent: str) -> list:
    rst = []
    for word in a_sent.split(' '):
        if word.strip() in ',./?!':
            continue
        rst.append(word.lower())
        # 불용어 처리 등등...
    return rst
