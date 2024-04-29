import torch
import torch.nn as nn


class StyleLoss(nn.Module):
    def __init__(self):
        super(StyleLoss, self).__init__()
        self.loss = nn.MSELoss()

    def gram_matrix(self, input):
        b, c, h, w, d = input.size()
        features = input.view(b * c, h * w * d)
        G = torch.mm(features, features.t())
        return G.div_(b * h * w * d)

    def forward(self, inputs, targets):
        G_inp = self.gram_matrix(inputs)
        G_target = self.gram_matrix(targets)
        style_loss = self.loss(G_inp, G_target)
        return style_loss


if __name__ == '__main__':
    x = torch.rand((2, 320, 4, 4, 8)) + 1
    x1 = torch.rand((2, 320, 4, 4, 8)) + 1
