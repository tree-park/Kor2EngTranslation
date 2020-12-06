import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from lib.data_preprocess import Vocab, TrainSet, collate_fn
from lib.model.seq2seq import Seq2Seq


class Translator:
    def __init__(self, dconf, mconf):
        self.dconf = dconf
        self.mconf = mconf

        # data pre-process & build data-loader
        self.ko_vocab = Vocab(self.dconf.min_cnt)
        self.en_vocab = Vocab(self.dconf.min_cnt)
        self.voc_size = 0
        self.dataset = None
        self._dataload = None

        # torch flow
        self.lm = None
        self.loss = None
        self.optim = None
        self.lrscheder = None

    def train(self):
        raise

    def save(self, fname):
        """ save model """
        torch.save(self.lm.state_dict(), '../results/model/' + fname)

    def load(self, path: str):
        """ load pytorch model """
        self.lm.load_state_dict(torch.load(path))

    def kor_to_eng(self, kor: str):
        eng = ''
        return eng

    def eng_to_kor(self, eng: str):
        kor = ''
        return kor


@torch.no_grad()
def accuracy(pred, target):
    acc = 0
    return acc

@torch.no_grad()
def perplexcity(pred, target):
    acc = 0
    return acc


class Seq2SeqModel(Translator):

    def train(self):
        self.dataset = TrainSet(self.dconf.train_ko_path, self.dconf.train_en_path, self.ko_vocab,
                                self.en_vocab)

        self._dataload = DataLoader(self.dataset,
                                    batch_size=self.mconf.batch_size,
                                    num_workers=0, collate_fn=collate_fn)
        print(len(self.ko_vocab), len(self.en_vocab))
        self.lm = Seq2Seq(len(self.ko_vocab) + 1, len(self.en_vocab) + 1,
                          self.mconf.emb_dim, self.mconf.hid_dim)
        self.loss = nn.NLLLoss()
        self.optim = optim.Adam(params=self.lm.parameters(), lr=self.mconf.lr)
        self.lrscheder = optim.lr_scheduler.ReduceLROnPlateau(self.optim, patience=5)

        for epoch in tqdm(range(self.mconf.epoch), desc='epoch'):
            total_loss = 0
            total_acc = 0

            self.lm.train()
            for i, batch in tqdm(enumerate(self._dataload), desc="step", total=len(self._dataload)):
                kor, end = batch
                self.optim.zero_grad()
                pred = self.lm(kor, end)
                pred, end = pred.view(-1, len(self.en_vocab) + 1), end.view(1, -1).squeeze(0)
                b_loss = self.loss(pred, end)
                b_loss.backward()
                self.optim.step()

                total_acc += accuracy(pred, end)
                total_loss -= b_loss.item()

            if epoch % 10 == 0:
                print()
                print(total_loss)


class LSTMAttention(Translator):

    def train(self):
        pass
