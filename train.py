
from lib.util import Config
from lib.kor2eng import LangTranslator
from lib.util import load_data
from lib.data_preprocess import Vocab, preprocessor
from lib.model.seq2seq import BiLSTMSeq2Seq
from transformers.lib.model.transformer import Transformer

# load configs
dconf_path = 'config/data.json'
mconf_path = 'config/lm.json'
dconf = Config(dconf_path)
mconf = Config(mconf_path)

# load & preprocess corpus
ko_corpus = preprocessor(load_data(dconf.train_ko_path), lang='ko')
en_corpus = preprocessor(load_data(dconf.train_en_path), lang='en')

# load vocab
ko_vocab = Vocab(dconf.min_cnt)
en_vocab = Vocab(dconf.min_cnt)
ko_vocab.load(ko_corpus)
en_vocab.load(en_corpus)

# define lm model
if mconf.model == 'seq2seq':
    model = BiLSTMSeq2Seq(len(ko_vocab) + 1, len(en_vocab) + 1,
                               mconf.emb_dim, mconf.d_m)
if mconf.model == 'transformer':
    model = Transformer(mconf.d_m, len(ko_vocab) + 1, len(en_vocab) + 1,
                        mconf.d_m*4, n=3)

# load translator and train
lm = LangTranslator(model, ko_vocab, en_vocab, dconf, mconf)
lm.train(ko_corpus, ko_corpus)

# save model
lm.save('trained.pth')
mconf.save(mconf_path)

test = ['또 하나 필요한 것은 훌륭한 영어 실력이다.',
        '경찰은 월요일 밤 집무실을 찾아 증거를 압수했다.']
print(lm.translate(test))
