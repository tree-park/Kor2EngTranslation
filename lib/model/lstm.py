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
        hidd = torch.cat((fw_hid[0], bw_hid[0]), dim=1)
        cell = torch.cat((fw_hid[1], bw_hid[1]), dim=1)
        return torch.transpose(output, 0, 1), (hidd, cell)


class LSTMLayer(nn.Module):
    def __init__(self, inp_size, hid_size):
        super().__init__()
        self.hid_size = hid_size
        self.enc_lstm_fw = nn.LSTMCell(inp_size, hid_size)

    def forward(self, inp):
        hidd = nn.init.xavier_normal_(torch.empty(inp.size(1), self.hid_size))
        cell = nn.init.xavier_normal_(torch.empty(inp.size(1), self.hid_size))
        outputs = []
        for i in range(inp.size(0)):
            hidd, cell = self.enc_lstm_fw(inp[i], (hidd, cell))
            outputs.append(hidd)
        return torch.stack(outputs), (hidd, cell)


class ReversedLSTMLayer(nn.Module):
    def __init__(self, inp_size, hid_size):
        super().__init__()
        self.hid_size = hid_size
        self.enc_lstm_fw = nn.LSTMCell(inp_size, hid_size)

    def forward(self, inp):
        
        hidd = nn.init.xavier_normal_(torch.empty(inp.size(1), self.hid_size))
        cell = nn.init.xavier_normal_(torch.empty(inp.size(1), self.hid_size))
        outputs = []
        for i in range(inp.size(0)-1, -1, -1):  
            
            hidd, cell = self.enc_lstm_fw(inp[i], (hidd, cell))
            outputs.append(hidd)
        return torch.stack(reverse(outputs)), (hidd, cell)


def reverse(lst):
    return lst[::-1]
