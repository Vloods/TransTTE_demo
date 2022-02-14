import torch.nn.functional as F
import torch.nn as nn


class FFNet(nn.Module):
    def __init__(self, input_size, norm_flag):
        
        super(FFNet, self).__init__()
        self.norm_flag = norm_flag
        self.norm = nn.LayerNorm(input_size, elementwise_affine=False)
        self.linear_1 = nn.Linear(input_size, 128)
        self.linear_2 = nn.Linear(128, 1)
        self.sftmax = nn.Softmax(dim=1) 
        
    def forward(self, x):
        if (self.norm_flag == True):
            x = self.norm(x)
        x = F.relu(self.linear_1(x))
        x = F.relu(self.linear_2(x))

        return x
