import torch.nn as nn
import numpy as np
import math


class Perplexity(nn.NLLLoss):
    MAX_THR = 100

    def eval_batch(self, pred, target):
        self.acc_loss += self.criterion(pred, target)
        if self.mask is None:
            self.norm_term += np.prod(target.size())
        else:
            self.norm_term += target.data.ne(self.mask).sum()

    def get_loss(self):
        loss = super(Perplexity, self).get_loss()
        loss /= self.norm_term.item()
        if loss > Perplexity.MAX_THR:
            print("Perplexity over 100 :", loss)
            return math.exp(Perplexity.MAX_THR)
        return math.exp(loss)
