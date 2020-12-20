import torch
import torch.nn as nn


class BiLSTM(nn.Module):

    def __init__(self, emb_size, hid_size):
        super().__init__()
        self.enc_lstm_fw = nn.LSTMCell(emb_size, hid_size)
        self.enc_lstm_bw = nn.LSTMCell(emb_size, hid_size)

    def forward(self, inp):
        """

        Args:
            inp ([maxlen, bsize, emb_size]):

        Returns:
            hid_out: [bsize, maxlen+1, emb_size*2]
            last_hid: [bsize, emb_size*2]
        """
        init_hidd = nn.init.xavier_normal_(torch.empty(inp.size(1), self.hid_size))
        init_cell = nn.init.xavier_normal_(torch.empty(inp.size(1), self.hid_size))

        fw_hidd = init_hidd  # [bsize, hid_size]
        fw_cell = init_cell  # [bsize, hid_size]
        fw_hid_states = []
        for i in range(inp.size(0)):  # maxlen
            # emb[i] [bsize, emb_size]
            fw_hidd, fw_cell = self.enc_lstm_fw(inp[i], (fw_hidd, fw_cell))
            fw_hid_states.append(fw_hidd)
        fw_hid_states.append(init_hidd)
        fw_hid_states = torch.stack(fw_hid_states)  # [maxlen+1, bsize, emb_size]

        bw_hid_states = []
        bw_hidd = init_hidd
        bw_cell = init_cell
        for i in range(inp.size(0), -1, -1):  # maxlen
            # emb[i] [bsize, emb_size] [bsize, emb_size, hid_size]
            bw_hidd, bw_cell = self.enc_lstm_bw(inp[i], (bw_hidd, bw_cell))
            bw_hid_states.append(bw_hidd)
        bw_hid_states.append(init_hidd)
        bw_hid_states = list(reversed(bw_hid_states))
        bw_hid_states = torch.stack(bw_hid_states)  # [maxlen+1, bsize, hid_size]

        hid_out = torch.cat((fw_hid_states, bw_hid_states), dim=2)  # [maxlen+1, bsize, hid_size*2]
        hid_out = torch.transpose(hid_out, 0, 1)  # [bsize, maxlen+1, hid_size*2]

        last_hidd = torch.cat((fw_hidd, bw_hidd), dim=2)  # [bsize, hid_size*2]
        last_cell = torch.cat((fw_hidd, bw_hidd), dim=2)  # [bsize, hid_size*2]
        return hid_out, (last_hidd, last_cell)
