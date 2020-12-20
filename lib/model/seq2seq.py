import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.model.lstm import BiLSTM
from lib.model.attention import ConcatAttention


class LSTMSeq2Seq(nn.Module):

    def __init__(self, ko_vocab_size, en_vocab_size, emb_size, hid_size):
        super().__init__()
        self.emb_size = emb_size
        self.hid_size = hid_size
        self.enc_emb = nn.Embedding(ko_vocab_size, emb_size)
        self.dec_emb = nn.Embedding(en_vocab_size, emb_size)
        self.enc_lstm_1 = nn.LSTMCell(emb_size, hid_size)
        self.enc_lstm_2 = nn.LSTMCell(hid_size, hid_size)
        self.dec_lstm_1 = nn.LSTMCell(emb_size, hid_size)
        self.dec_lstm_2 = nn.LSTMCell(hid_size, hid_size)
        self.out_layer = nn.Linear(hid_size, en_vocab_size)

    def encoder(self, inp):
        # inp [batchsize, maxlen]
        # emb [batchsize, maxlen, emb_size] => [maxlen, batchsize, emb_size]
        # emb = self.enc_emb(inp).view(inp.size(1), inp.size(0), -1)
        emb = torch.transpose(self.enc_emb(inp), 0, 1)
        # hid_state [batch_size, hid_size]
        hid_state = torch.empty(inp.size(0), self.hid_size)
        cell_state = torch.empty(inp.size(0), self.hid_size)
        hid_state = nn.init.xavier_normal_(hid_state)
        cell_state = nn.init.xavier_normal_(cell_state)
        for i in range(inp.size(1)):  # maxlen
            # emb[i] [batchsize, emb_size] [emb_size, hid_size]
            hid_state, cell_state = self.enc_lstm_1(emb[i], (hid_state, cell_state))
            hid_state, cell_state = self.enc_lstm_2(hid_state, (hid_state, cell_state))
        return hid_state, cell_state

    def decoder(self, enc_hid, enc_cel, inp):
        hid_state = enc_hid
        cell_state = enc_cel
        # emb [batchsize, maxlen, emb_size] => [maxlen, batchsize, emb_size]
        # dec_emb = self.dec_emb(inp).view(inp.size(1), inp.size(0), -1)
        dec_emb = torch.transpose(self.dec_emb(inp), 0, 1)
        out = []
        for i in range(0, inp.size(1)):
            hid_state, cell_state = self.dec_lstm_1(dec_emb[i], (hid_state, cell_state))
            hid_state, cell_state = self.dec_lstm_2(hid_state, (hid_state, cell_state))
            out.append(hid_state)
        out = torch.transpose(torch.stack(out), 0, 1)
        out = F.relu(self.out_layer(out))
        out = F.log_softmax(out, dim=2)
        return out

    def forward(self, inp, tar):
        enc_hid, enc_cell = self.encoder(inp)
        dec = self.decoder(enc_hid, enc_cell, tar)
        return F.log_softmax(dec)


class BiLSTMSeq2Seq(nn.Module):

    def __init__(self, ko_vocab_size, en_vocab_size, emb_size, hid_size):
        super().__init__()
        self.emb_size = emb_size
        self.hid_size = hid_size
        self.enc_emb = nn.Embedding(ko_vocab_size, emb_size)
        self.dec_emb = nn.Embedding(en_vocab_size, emb_size)
        self.bi_lstm = BiLSTM(emb_size, hid_size)
        self.attn = ConcatAttention(hid_size)
        self.dec_lstm = nn.LSTMCell(emb_size + hid_size * 2, hid_size * 2)
        self.out_layer = nn.Linear(hid_size * 2, en_vocab_size)

    def encoder(self, inp):
        """

        Args:
            inp ([bsize, maxlen]):

        Returns:
            output ([bsize, maxlen+1, hid_size*2])
            hid ([bsize, hsize*2])

        """
        # emb [batchsize, maxlen, emb_size] => [maxlen, batchsize, emb_size]
        emb = torch.transpose(self.enc_emb(inp), 0, 1)
        output, hid = self.bi_lstm(emb)
        return output, hid

    def decoder(self, inp, enc_output, enc_last_hid):
        """
        Teacher forcing Decoder
        Args:
            enc_output ([bsize, maxlen, hid_size*2]):
            enc_last_hid ([batch_size, hid_size*2]):
            inp ([bsize, maxlen]):

        Returns:

        """
        hid_state, cell_state = enc_last_hid

        # dec_emb [batchsize, maxlen, emb_size] => [maxlen, batchsize, emb_size]
        dec_emb = torch.transpose(self.dec_emb(inp), 0, 1)
        out = []
        for i in range(0, inp.size(1)):
            # Cal attention
            cvec = self.attn(hid_state, enc_output)  # [bsize, hid_size*2]
            # emb : [batchsize, emb_size + hid_size*2]
            emb = torch.cat((dec_emb[i], cvec), dim=1)
            hid_state, cell_state = self.dec_lstm(emb, (hid_state, cell_state))
            out.append(hid_state)
        out = torch.transpose(torch.stack(out), 0, 1)
        out = F.relu(self.out_layer(out))
        out = F.log_softmax(out, dim=2)
        return out

    def forward(self, inp, tar):
        """

        Args:
            inp ():
            tar ():

        Returns:

        """
        output, hid = self.encoder(inp)
        dec = self.decoder(tar, output, hid)
        return dec

    def predict(self, inp, maxlen):
        """

        Args:
            inp ():
            maxlen ():

        Returns:

        """
        enc_output, (hid_state, cell_state) = self.encoder(inp)
        out = []
        dec_inp = inp  # [batch]
        for i in range(0, maxlen):
            # Cal attention
            cvec = self.attn(hid_state, enc_output)  # [bsize, hid_size*2]
            # emb : [batchsize, emb_size + hid_size*2]
            dec_emb = self.dec_emb(dec_inp)  # [bsize, emb]
            emb = torch.cat((dec_emb, cvec), dim=1)
            hid_state, cell_state = self.dec_lstm(emb, (hid_state, cell_state))

            dec_inp = torch.argmax(F.log_softmax(F.relu(self.out_layer(emb)), dim=1))  # [batch]
            out.append(dec_inp)
        out = torch.transpose(torch.stack(out), 0, 1)
        return out


class LSTMSeq2Seq2(nn.Module):
    def __init__(self, ko_vocab_size, en_vocab_size, emb_size, hid_size):
        super().__init__()
        self.enc_emb = nn.Embedding(ko_vocab_size, emb_size)
        self.dec_emb = nn.Embedding(en_vocab_size, emb_size)
        self.enc_lstm = nn.LSTM(input_size=emb_size, hidden_size=hid_size, num_layers=2,
                                batch_first=True)
        self.dec_lstm = nn.LSTM(input_size=emb_size, hidden_size=hid_size, num_layers=2,
                                batch_first=True)
        self.out_layer = nn.Linear(hid_size, en_vocab_size, bias=True)

    def encoder(self, inp):
        # inp [batchsize, maxlen]
        # emb [batchsize, maxlen, emb_size]
        enc_emb = self.enc_emb(inp)
        # hid_state [emb_size, hid_size]
        enc, (h, c) = self.enc_lstm(enc_emb)
        # enc [batchsize, maxlen, hid_size]
        return h, c

    def decoder(self, enc_hid, enc_cel, inp):
        # inp [batchsize, maxlen]
        # dec_emb [batchsize, maxlen, emb_size]
        dec_emb = self.dec_emb(inp)
        # out [batchsize, maxlen, hid_size]
        out, _ = self.dec_lstm(dec_emb, (enc_hid, enc_cel))
        # out [batchsize, maxlen, en_vocab_size]
        out = F.relu(self.out_layer(out))
        out = F.log_softmax(out, dim=2)
        return out

    def forward(self, inp, tar):
        enc_hid, enc_cell = self.encoder(inp)
        dec = self.decoder(enc_hid, enc_cell, tar)
        return F.log_softmax(dec)
