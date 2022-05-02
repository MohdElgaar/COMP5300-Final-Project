import torch
import torch.nn as nn

class DPWeighting(nn.Module):
    def __init__(self, tao, alpha = 0.5):
        super().__init__()
        self.tao = tao
        self.alpha = alpha

    def forward(self, loss, diff, *args):
        confs = torch.where(diff >= self.tao,
                1 - self.alpha * (diff - self.tao) / (1 - self.tao),
                1.0)

        return confs
