import torch
from torch.utils.data import Dataset


def load_data(path: str):
    with open(path, 'r') as f:
        result = f.readlines()
    return result


class Corpus(Dataset):

    def __init__(self, data_set):
        self._data = data_set

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx: int):
        """ return input word and target word """
        ko, en = self._data[idx]
        return torch.tensor(ko), torch.tensor(en)


def collate_fn(batch):
    ko, en = zip(*batch)
    pad_ko = torch.nn.utils.rnn.pad_sequence(ko,batch_first=True)
    pad_en = torch.nn.utils.rnn.pad_sequence(en,batch_first=True)
    return pad_ko, pad_en