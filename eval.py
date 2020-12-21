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

corpus = """
특수 부대가 극장 건물로 공격해 들어가면서, 러시아인이 아닌 75명의 외국인과 25명의 어린이 전부를 포함해 750명 이상의 인질들이 구출되었다.
러시아 특수 부대는 극장으로 공격해 들어가기 전에 신경 독가스를 사용했다.
많은 인질들이 화학 가스의 영향으로 고통을 겪으며 병원으로 옮겨졌다.
과거 노조 지도자였던 루이스 이나시오 룰라 다 실바가 브라질의 대통령 선거 결선 투표에서 압승을 거두면서, 라틴 아메리카 최대 국가로서는 좌파로의 역사적인 전환을 맞이하게 되었다.
실바의 노동자당이 자신들의 후보가 선거의 승리자라고 선언한 후 몇 시간이 지나, 집권당 후보인 조제 세하는 패배를 시인했다.
수천명의 실바 지지자들이 상 파울로와 리오 데 자네이로의 거리에 모여 들어, 실바가 이끄는 당의 붉은 깃발을 흔들며 축하했다.
북한의 핵무기 계획을 포기하도록 하려는 압력이 거세지고 있는 가운데, 일본과 북한의 외교관들이 외교 관계를 정상화하려는 회담을 재개했다.
말레이시아의 수도 쿠알라 룸푸르에서 열리는 이틀간의 회담에서는 1970년대와 80년대에 있었던 일본 민간인들의 납치에 대한 보다 충분한 설명을 요청하는 것 뿐만아니라 북한의 (핵)폭탄 제조 노력에 대한 일본의 우려가 주로 논의될 것으로 예상되고 있다.
100여명의 사망자가 예상되는 베트남 빌딩 화재
"""
test = corpus.split('\n')

print(lm.translate(test))
