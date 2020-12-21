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
lm.load('trained.pth')

test = ['또 하나 필요한 것은 훌륭한 영어 실력이다.', '또 하나 필요한 것은 훌륭한 영어 실력이다.']

print(lm.translate(test))
