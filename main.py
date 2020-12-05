"""
download data from
https://github.com/j-min/korean-parallel-corpora/tree/master/korean-english-v1

"""

from lib.util import Config
from lib.kor2eng import Translator


# load configs
dconf_path = 'config/data.json'
mconf_path = 'config/word2vec.json'
dconf = Config(dconf_path)
mconf = Config(mconf_path)

# load w2v model and train
lm = Translator(dconf, mconf)
lm.train()
