import torch
import torch.nn as nn
import torch.nn.functional as F


class ConcatAttention(nn.Module):
    """ Concat Attention """
    def __init__(self, hid_size):
        super(ConcatAttention, self).__init__()
        self.align = nn.Linear(hid_size * 2, hid_size)
        self.v = nn.Linear(hid_size, 1)

    def forward(self, dec_hid, enc_hids):
        """
        General Attention mechanism
        Args:
            dec_hid ([batch_size, hid_size*2]):
            enc_hids ([bsize, maxlen, hid_size*2]):

        Returns:
            context_vec: [bsize, hid_size*2]
        """
        # 1. vatten 유사도 = enc_hid * dec_hid : [bsize, maxlen, hid_size*2] *[batch_size, hid_size*2]
        dec_hid = dec_hid.unsqueeze(1).expand(-1, enc_hids.size(1), -1)  # [batch_size, maxlen, hid_size*2]
        energy = F.tanh(self.align(torch.cat((dec_hid, enc_hids), dim=2)))  # [batch_size, maxlen, hid_size]
        energy = self.v(energy).squeeze()  # [bsize, maxlen]
        attn = F.softmax(energy).unsqueeze(1)  # [bsize, 1, maxlen]
        context_vec = torch.bmm(attn, enc_hids).squeeze()  # [bsize, 1, maxlen] * [bsize, maxlen, hid_size*2]
        return context_vec


class DotAttention(nn.Module):
    """ Dot Product Attention """
    def __init__(self, hid_size, scaled=True):
        super(DotAttention, self).__init__()
        self.scaled = scaled

    def forward(self, dec_hid, enc_hids):
        """
        General Attention mechanism
        Args:
            dec_hid ([batch_size, hid_size*2]):
            enc_hids ([bsize, maxlen, hid_size*2]):

        Returns:
            context_vec: [bsize, hid_size*2]
        """
        # [bsize, maxlen, hid_size*2] * [batch_size, hid_size*2]
        attn = torch.bmm(enc_hids, dec_hid)  # [bsize, maxlen]
        if self.scaled:
            attn = attn * (1/torch.sqrt(dec_hid.size(-1)))
        attn = F.softmax(attn, dim=1)
        # [bsize, maxlen] * [bsize, maxlen, hid_size*2]
        context_vec = torch.matmul(attn, enc_hids)  # [bsize, hid_size*2]
        return context_vec
