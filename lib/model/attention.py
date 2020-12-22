import torch
import torch.nn as nn
import torch.nn.functional as F


class ConcatAttention(nn.Module):
    """ Concat Attention """

    def __init__(self, hid_size):
        super(ConcatAttention, self).__init__()
        self.align = nn.Linear(hid_size * 4, hid_size)
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

        dec_hid = dec_hid.unsqueeze(1).expand(-1, enc_hids.size(1), -1)
        energy = F.tanh(self.align(torch.cat((dec_hid, enc_hids), dim=2)))
        energy = self.v(energy).squeeze()
        attn = F.softmax(energy).unsqueeze(1)
        context_vec = torch.bmm(attn, enc_hids).squeeze()
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

        attn = torch.bmm(enc_hids, dec_hid)
        if self.scaled:
            attn = attn * (1 / torch.sqrt(dec_hid.size(-1)))
        attn = F.softmax(attn, dim=1)

        context_vec = torch.matmul(attn, enc_hids)
        return context_vec
