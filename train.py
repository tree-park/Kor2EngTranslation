# -*- coding: utf-8 -*-
import os
import sys
import torch
import pickle

from lib.util import Config
from lib.kor2eng import LangTranslator
from lib.util import load_data
from lib.data_preprocess import Vocab, preprocessor
from lib.model.seq2seq import BiLSTMSeq2Seq
from transformer.lib.model.transformer import Transformer

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

# load configs
dconf_path = 'config/data.json'
mconf_path = 'config/lm.json'
dconf = Config(dconf_path)
mconf = Config(mconf_path)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print('Using device:', device)

try:
    with open('preprocessed_data.pickle', 'rb') as f:
        saved_obj = pickle.load(f)
        ko_corpus, ko_vocab, en_corpus, en_vocab = saved_obj
except:

    # load & preprocess corpus
    ko_corpus = preprocessor(load_data(dconf.train_ko_path), lang='ko')
    en_corpus = preprocessor(load_data(dconf.train_en_path), lang='en')

    # load vocab
    ko_vocab = Vocab(dconf.min_cnt)
    en_vocab = Vocab(dconf.min_cnt)
    ko_vocab.load(ko_corpus)
    en_vocab.load(en_corpus)

    # save data as pickle
    with open('preprocessed_data.pickle', 'wb') as f:
        pickle.dump([ko_corpus, ko_vocab, en_corpus, en_vocab], f, protocol=pickle.HIGHEST_PROTOCOL)

assert all([ko_corpus, ko_vocab, en_corpus, en_vocab])

# define lm model
if mconf.model == 'transformer':
    model = Transformer(mconf.d_m, len(ko_vocab) + 1, len(en_vocab) + 1,
                        mconf.d_m * 8, n=6)
else:
    model = BiLSTMSeq2Seq(len(ko_vocab) + 1, len(en_vocab) + 1,
                          mconf.emb_dim, mconf.d_m)
model.to(device)

# load translator and train
lm = LangTranslator(model, ko_vocab, en_vocab, dconf, mconf, device)
lm.train(ko_corpus, en_corpus)

# save model
lm.save('trained.pth')
mconf.save(mconf_path)

test = ['또 하나 필요한 것은 훌륭한 영어 실력이다.',
        '경찰은 월요일 밤 집무실을 찾아 증거를 압수했다.']
print(lm.translate(test))
