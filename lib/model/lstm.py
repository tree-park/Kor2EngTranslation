import torch
import torch.nn as nn


class BiLSTM(nn.Module):
    def __init__(self, inp_size, hid_size):
        super(BiLSTM, self).__init__()
        self.fw_lstm = LSTMLayer(inp_size, hid_size)
        self.bw_lstm = ReversedLSTMLayer(inp_size, hid_size)

    def forward(self, inp):
        fw_out, fw_hid = self.fw_lstm(inp)
        bw_out, bw_hid = self.bw_lstm(inp)
        output = torch.cat((fw_out, bw_out), dim=2)
        hid_stat = torch.cat((fw_hid, bw_hid), dim=2)
        return output, hid_stat


class LSTMLayer(nn.Module):
    def __init__(self, inp_size, hid_size):
        super().__init__()
        self.enc_lstm_fw = nn.LSTMCell(inp_size, hid_size)

    def forward(self, inp):
        # [bsize, hid_size]
        hidd = nn.init.xavier_normal_(torch.empty(inp.size(1), self.hid_size))
        cell = nn.init.xavier_normal_(torch.empty(inp.size(1), self.hid_size))
        outputs = []
        for i in range(inp.size(0)):  # maxlen
            # emb[i] [bsize, emb_size]
            hidd, cell = self.enc_lstm_fw(inp[i], (hidd, cell))
            outputs.append(hidd)
        return torch.stack(outputs), (hidd, cell)


class ReversedLSTMLayer(nn.Module):
    def __init__(self, inp_size, hid_size):
        super().__init__()
        self.enc_lstm_fw = nn.LSTMCell(inp_size, hid_size)

    def forward(self, inp):
        # [bsize, hid_size]
        hidd = nn.init.xavier_normal_(torch.empty(inp.size(1), self.hid_size))
        cell = nn.init.xavier_normal_(torch.empty(inp.size(1), self.hid_size))
        outputs = []
        for i in range(inp.size(0), -1, -1):  # maxlen reversed
            # emb[i] [bsize, emb_size]
            hidd, cell = self.enc_lstm_fw(inp[i], (hidd, cell))
            outputs.append(hidd)
        return torch.stack(reverse(outputs)), (hidd, cell)


def reverse(lst):
    return lst[::-1]
