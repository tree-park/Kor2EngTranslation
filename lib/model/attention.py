import torch
import torch.nn as nn

class Attention(nn.Module):

    def __init__(self):
        pass
        # Linear fc 생성 for attention weights
        # softmax

    def forward(self, enc_hid, dec_hid):
        pass
        # 1. vatten 유사도 = enc_hid * dec_hid [bsize, maxlen, hid*2] * [bsize, maxlen, emb]
        # 2. atten distribution = fc(atten 유사도)
        # 3. softmax
        # 4. fc((3) * enc_hid)