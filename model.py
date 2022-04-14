
import numpy as np
import torch
import torch.nn as nn
from utils import sigmoid, d_sigmoid

class MLP:
    
    def __init__(self):

        self.w1 = np.matrix(([1.0, 2.0], [0.3, 0.4], [2.0, 1.0]))
        self.w2 = np.matrix(([1.0, 0.5, 2.0], [2.0, 1.0, 1.0]))
        self.w1_init = np.matrix(([1.0, 2.0], [0.3, 0.4], [2.0, 1.0]))
        self.w2_init = np.matrix(([1.0, 0.5, 2.0], [2.0, 1.0, 1.0]))

        self.z1 = None
        self.a1 = None
        self.z2 = None
        self.a2 = None
        
        self.y_pred = None

        self.grad_w1 = None
        self.grad_w2 = None

        self.curr_loss = None

    def forward(self, x):
        # x: 2x1, w1: 3x2
        self.z1 = np.dot(self.w1, x)
        # z1: 3x1
        self.a1 = sigmoid(self.z1)
        # a1: 3x1, w2: 2x3
        self.z2 = np.dot(self.w2, self.a1)
        # z2: 2x1
        self.a2 = self.z2
        # a2: 2x1
        self.y_pred = self.a2
    
    def get_loss(self, y):

        self.curr_loss = np.sum(0.5 * (np.array(y) - np.array(self.y_pred)) ** 2)
    

    def back_propagation(self, x, y):

        grad_a2 = self.y_pred - y
        grad_z2 = grad_a2
        self.grad_w2 = np.dot(grad_z2, self.a1.T)

        grad_a1 = np.dot(self.w2.T, grad_z2)
        grad_z1 = np.multiply(d_sigmoid(self.z1), grad_a1)
        self.grad_w1 = np.dot(grad_z1, x.T)

    def update_w(self, lr):

        self.w1 -= lr * self.grad_w1
        self.w2 -= lr * self.grad_w2


class MLPTorch(nn.Module):
    
    def __init__(self):
        super(MLPTorch, self).__init__()
        self.fc1 = nn.Linear(2,3,bias=False)
        self.fc2 = nn.Linear(3,2,bias=False)
        self.sigmoid = nn.Sigmoid()
        self.fc1.weight = nn.Parameter(torch.tensor([[1,2],[0.3,0.4],[2,1]]))
        self.fc2.weight = nn.Parameter(torch.tensor([[1,0.5,2],[2,1,1]]))

    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        return x