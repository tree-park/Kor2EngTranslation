import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import math

from lib.data_handle import load_data
from lib.data_preprocess import Vocab, Corpus, collate_fn, preprocessor
from lib.model.seq2seq import LSTMSeq2Seq, LSTMSeq2Seq2, BiLSTMSeq2Seq


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
        self.model = None
        self.loss = None
        self.perpelexity = None
        self.optim = None
        self.lrscheder = None

    def train(self):
        raise

    def predict(self, corpus):
        raise

    def save(self, fname: str):
        """ save model """
        # torch.save(self.model.state_dict(), 'results/model/' + fname)
        torch.save({
            'model': self.model.state_dict(),
            'optim': self.optim.state_dict(),
            'ko_vocab': self.ko_vocab,
            'en_vocab': self.en_vocab
        }, 'results/model/' + fname)

    def load(self, fname: str, retrain=False):
        """ load pytorch model """
        if not self.model:
            raise
        checkpoint = torch.load('results/model/' + fname)
        self.model.load_state_dict(checkpoint['model'])
        if self.optim and retrain:
            self.optim.load_state_dict(checkpoint['optim'])
        self.ko_vocab = checkpoint['ko_vocab']
        self.en_vocab = checkpoint['en_vocab']
        self.en_vocab.to_idx2word()
        self.model.eval()
        print(len(self.ko_vocab), len(self.en_vocab))

    def translate(self, kor: list):
        """ Translate Korean to English """
        pred = self.predict(kor)
        print(pred)
        # convert predict to en wid
        rst = []
        for sent_idx in pred:
            sent = [self.en_vocab.get_word(idx) for idx in sent_idx if not 0]
            rst.append(sent)
        return rst


@torch.no_grad()
def accuracy(pred, target):
    acc = sum(pred.argmax(1) == target).item() / len(target)
    return acc


class Seq2SeqModel(Translator):

    def train(self):
        ko_corpus = preprocessor(load_data(self.dconf.train_ko_path))
        en_corpus = preprocessor(load_data(self.dconf.train_en_path))
        self.ko_vocab.load(ko_corpus)
        self.en_vocab.load(en_corpus)

        train_set = self.trainset_form(ko_corpus, en_corpus, self.ko_vocab, self.en_vocab)
        self.dataset = Corpus(train_set)

        self._dataload = DataLoader(self.dataset,
                                    batch_size=self.mconf.batch_size,
                                    num_workers=0, collate_fn=collate_fn)
        print(len(self.ko_vocab), len(self.en_vocab))
        self.mconf.ko_size, self.mconf.en_size = len(self.ko_vocab) + 1, len(self.en_vocab) + 1
        # self.mconf[]
        # self.lm = LSTMSeq2Seq(len(self.ko_vocab) + 1, len(self.en_vocab) + 1,
        #                       self.mconf.emb_dim, self.mconf.hid_dim)
        self.model = BiLSTMSeq2Seq(len(self.ko_vocab) + 1, len(self.en_vocab) + 1,
                                   self.mconf.emb_dim, self.mconf.hid_dim)
        self.loss = nn.CrossEntropyLoss()
        self.optim = optim.Adam(params=self.model.parameters(), lr=self.mconf.lr)
        self.lrscheder = optim.lr_scheduler.ReduceLROnPlateau(self.optim, patience=5)

        total_loss = 0
        total_acc = 0
        self.model.train()
        self.info()
        for epoch in tqdm(range(self.mconf.epoch), desc='epoch'):
            for i, batch in tqdm(enumerate(self._dataload), desc="step", total=len(self._dataload)):
                ko, en = batch
                self.optim.zero_grad()
                en_xs = en[:, :-1]
                en_ts = en[:, 1:]
                pred = self.model(ko, en_xs)
                pred, en_ts = pred.view(-1, pred.shape[2]), en_ts.reshape(1, -1).squeeze(0)
                b_loss = self.loss(pred, en_ts)
                b_loss.backward()
                self.optim.step()

                total_acc += accuracy(pred, en_ts)
                total_loss += b_loss.item()

            # if epoch % 10 == 0:
            #     ppl = math.exp(total_loss/10)
            #     total_acc = 0
            itersize = math.ceil(len(self.dataset) / self.mconf.batch_size)
            ppl = math.exp(total_loss / itersize)
            print(epoch, total_loss, total_acc / itersize, ppl)
            self.lrscheder.step(total_loss)
            total_loss = 0
        self.en_vocab.to_idx2word()

    def load(self, fname: str, retrain=False):
        self.model = BiLSTMSeq2Seq(self.mconf.ko_size, self.mconf.en_size,
                                   self.mconf.emb_dim, self.mconf.hid_dim)
        Translator.load(self, fname)

    def predict(self, corpus):
        ko_corpus = preprocessor(corpus)
        pred_set = self.predset_form(ko_corpus, self.ko_vocab)
        dataset = torch.tensor(pred_set)
        dataset = torch.nn.utils.rnn.pad_sequence(dataset, batch_first=True)
        pred = self.model.predict(dataset, maxlen=dataset.size(1))
        return pred

    def trainset_form(self, ko_corpus, en_corpus, ko_vocab, en_vocab):
        rst = []
        for ko, en in zip(ko_corpus, en_corpus):
            ko = [ko_vocab[x] for x in ko]
            en = [en_vocab[x] for x in en]
            # padding
            rst.append([ko, en])
        return rst

    def predset_form(self, corpus, vocab):
        rst = []
        for ko in corpus:
            ko = [vocab[x] for x in ko]
            # padding
            rst.append(ko)
        return rst

    def info(self):
        # Print model's state_dict
        print("Model's state_dict:")
        for param_tensor in self.model.state_dict():
            print(param_tensor, "\t", self.model.state_dict()[param_tensor].size())

        # Print optimizer's state_dict
        print("Optimizer's state_dict:")
        for var_name in self.optim.state_dict():
            print(var_name, "\t", self.optim.state_dict()[var_name])
