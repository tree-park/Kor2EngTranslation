"""
download data from
https://github.com/j-min/korean-parallel-corpora/tree/master/korean-english-v1

"""

from lib.util import Config
from lib.kor2eng import Seq2SeqModel


# load configs
dconf_path = 'config/data.json'
mconf_path = 'config/lm.json'
dconf = Config(dconf_path)
mconf = Config(mconf_path)

# load w2v model and train
lm = Seq2SeqModel(dconf, mconf)
lm.train()
lm.save('trained')
