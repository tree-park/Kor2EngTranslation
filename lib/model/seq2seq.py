import torch
import torch.nn as nn
import torch.nn.functional as F


class Seq2Seq(nn.Module):

    def __init__(self, ko_vocab_size, en_vocab_size, emb_size, hid_size):
        super().__init__()
        self.emb_size = emb_size
        self.hid_size = hid_size
        self.enc_emb = nn.Embedding(ko_vocab_size, hid_size)
        self.dec_emb = nn.Embedding(en_vocab_size, hid_size)
        self.enc_lstm = nn.LSTMCell(hid_size, hid_size)
        self.dec_lstm = nn.LSTMCell(hid_size, hid_size)
        self.out_layer = nn.Linear(hid_size, en_vocab_size)

    def encoder(self, inp):
        # inp [batchsize, maxlen]
        # emb [batchsize, maxlen, emb_size] => [maxlen, batchsize, emb_size]
        # emb = self.enc_emb(inp).view(inp.size(1), inp.size(0), -1)
        emb = torch.transpose(self.enc_emb(inp), 0, 1)
        # hid_state [hid_size, hid_size]
        hid_state = torch.zeros(inp.size(0), self.hid_size)
        cell_state = torch.zeros(inp.size(0), self.hid_size)
        for i in range(inp.size(1)):
            # [batchsize, emb_size] [hid_size, hid_size]
            hid_state, cell_state = self.enc_lstm(emb[i], (hid_state, cell_state))
        return hid_state, cell_state

    def decoder(self, enc_hid, enc_cel, inp):
        hid_state = enc_hid
        cell_state = enc_cel
        # emb [batchsize, maxlen, emb_size] => [maxlen, batchsize, emb_size]
        # dec_emb = self.dec_emb(inp).view(inp.size(1), inp.size(0), -1)
        dec_emb = torch.transpose(self.dec_emb(inp), 0, 1)
        out = dec_emb[0].unsqueeze(0)
        for i in range(1, inp.size(1)):
            hid_state, cell_state = self.dec_lstm(dec_emb[i], (hid_state, cell_state))
            out = torch.cat([out, hid_state.unsqueeze(0)], dim=0)
        out = torch.transpose(out, 0, 1)
        out = F.relu(self.out_layer(out))
        out = F.log_softmax(out, dim=2)
        return out

    def forward(self, inp, tar):
        enc_hid, enc_cell = self.encoder(inp)
        dec = self.decoder(enc_hid, enc_cell, tar)
        return F.log_softmax(dec)
