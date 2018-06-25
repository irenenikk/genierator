import torch
import torch.nn as nn
import torch.nn.functional as func

class Gen(nn.Module):
    def __init__(self):
        super(Gen, self). __init__()
        # input is a vector of random numbers
        # output is a fake image (pixels)
        # tanh as an activation function?
        self.input_size = 100
        self.l1 = nn.Linear(self.input_size, 128)
        self.l2 = nn.Linear(128,512)
        self.l3 = nn.Linear(512, 784)

    def forward(self, x):
        x = x.view((x.size(0), -1))
        x = func.relu(self.l1(x))
        x = func.relu(self.l2(x))
        return func.tanh(self.l3(x))
    
    def loss_fun(self):
        # binary classification
        return nn.BCELoss()
    
class Discr(nn.Module):
    def __init__(self):
        super(Discr, self). __init__()
        # input is a vector of pixels
        # output is 0 (fake) or 1 (real)
        self.l1 = nn.Linear(784, 512)
        self.l2 = nn.Linear(512, 128)
        self.l3 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.l1(x)
        x = func.relu(self.l2(x))
        return func.softmax(self.l3(x))

    def loss_fun(self):
        # binary classification
        return nn.BCELoss()
    