import torch
import torch.nn as nn
import torch.nn.functional as F


class Seq2Seq(nn.Module):

    def __init__(self, vocab_size, emb_size, hid_size, out_size):
        super().__init__()
        self.hid_size = hid_size
        self.emb = nn.Embedding(vocab_size, emb_size)
        self.enc_lstm = nn.LSTMCell(emb_size, hid_size)
        self.dec_lstm = nn.LSTMCell(hid_size, out_size)

    def encoder(self, inp):
        hid_state = torch.zeros(inp.size(0), self.hid_size)
        cell_state = torch.zeros(inp.size(0), self.hid_size)
        rst = []
        emb = self.emb(inp).view(inp.size(1), inp.size(0), -1)

        for i in range(emb.size(0)):
            hid_state, cell_state = self.enc_lstm(emb[i], (hid_state, cell_state))
        return hid_state

    def decoder(self, enc):
        hid_state = torch.zeros(enc.size(0), self.hid_size)
        cell_state = torch.zeros(enc.size(0), self.hid_size)
        out = []
        for i in range(enc.size(0)):
            hid_state, cell_state = self.dec_lstm(enc[i], (hid_state, cell_state))
            out.append(hid_state)
        return out

    def forward(self, inp):
        enc_hid = self.encoder(inp)
        # dec = self.decoder(enc[-1])
        return F.log_softmax(enc_hid)
