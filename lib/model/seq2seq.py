import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.enc_lstm_fw = nn.LSTMCell(emb_size, hid_size)
        self.enc_lstm_bw = nn.LSTMCell(hid_size, hid_size)
        self.dec_lstm_1 = nn.LSTMCell(emb_size, hid_size)
        self.dec_lstm_2 = nn.LSTMCell(hid_size, hid_size)
        self.out_layer = nn.Linear(hid_size, en_vocab_size)

    def encoder(self, inp):
        # inp [batchsize, maxlen]
        # emb [batchsize, maxlen, emb_size] => [maxlen, batchsize, emb_size]
        # emb = self.enc_emb(inp).view(inp.size(1), inp.size(0), -1)
        emb = torch.transpose(self.enc_emb(inp), 0, 1)
        # hid_state [batch_size, hid_size]
        init_hidd = torch.empty(inp.size(0), self.hid_size)
        init_cell = torch.empty(inp.size(0), self.hid_size)
        init_hidd = nn.init.xavier_normal_(init_hidd)
        init_cell = nn.init.xavier_normal_(init_cell)

        hid_state = init_hidd
        cell_state = init_cell
        fw_hid_states = []
        for i in range(inp.size(1)):  # maxlen
            # emb[i] [batchsize, emb_size] [emb_size, hid_size]
            hid_state, cell_state = self.enc_lstm_fw(emb[i], (hid_state, cell_state))
            fw_hid_states.append(hid_state)
        fw_hid_states.append(init_hidd)
        fw_hid_states = torch.stack(fw_hid_states)

        # fw_out = torch.transpose(torch.stack(fw_hid_states), 0, 1)
        bw_hid_states = []
        hid_state = init_hidd
        cell_state = init_cell
        for i in range(inp.size(1), -1, -1):  # maxlen
            # emb[i] [batchsize, emb_size] [emb_size, hid_size]
            hid_state, cell_state = self.enc_lstm_bw(hid_state, (hid_state, cell_state))
            bw_hid_states.append(hid_state)
        bw_hid_states.append(init_hidd)
        bw_hid_states = list(reversed(bw_hid_states))
        bw_hid_states = torch.stack(bw_hid_states)

        hid_out = torch.cat((fw_hid_states, bw_hid_states), dim=2)
        hid_out = torch.transpose(hid_out, 0, 1)
        return hid_out, cell_state

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
