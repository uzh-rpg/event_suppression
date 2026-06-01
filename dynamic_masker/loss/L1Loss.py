import torch.nn as nn

class L1Loss(nn.Module):
    def __init__(self, **kwargs):
        super(L1Loss, self).__init__()
        self.loss = nn.L1Loss()

    def __call__(self, pred, target):
        return self.loss(pred, target)