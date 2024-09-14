import torch
import torch.nn as nn
import torch.nn.functional as F

class SE3Net(nn.Module):
    def __init__(self):
        super(SE3Net, self).__init__()
        self.fc1 = nn.Linear(6, 100)
        self.fc2 = nn.Linear(6, 100)
        self.fc3 = nn.Linear(200, 100)
        self.fc4 = nn.Linear(100, 50)
        self.fc5 = nn.Linear(50, 32)
    def forward(self, x):
        x=x.to(torch.float32)
        x1 = F.relu(self.fc1(x))
        x2 = F.relu(self.fc2(x))
        x3 = torch.cat((x1, x2), dim=1)
        x = self.fc3(x3)
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x