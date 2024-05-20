import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
 
    def forward(self, pred, mask):
        """
        :param pred: softmax(pred)
        :param mask: one_hot(mask)
        :return:
        """
        eps = 1e-7
        p = pred.view((pred.size()[0], pred.size()[1], -1))
        y = mask.view(p.size())
 
        ce = -1 * torch.log(p + eps) * y
        floss = torch.pow((1-p), self.gamma) * ce
        floss = torch.mul(floss, self.alpha)
        floss = torch.sum(floss, dim=1)
        return torch.mean(floss)
