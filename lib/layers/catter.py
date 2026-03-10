import torch

# concatenator

class Catter(torch.nn.Module):
    '''concatenates along the final dimension'''
    def __init__(self):
        super(Catter, self).__init__()

    def forward(self, x):
        return torch.cat(x, dim=-1)