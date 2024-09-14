import torch
import torch.nn as nn
import torch.nn.functional as F

class ufunc(nn.Module):
    def __init__(self):
        super(ufunc, self).__init__()
        self.fc1 = nn.Linear(32, 100)
        self.fc2 = nn.Linear(32, 100)
        self.fc3 = nn.Linear(200, 100)
        self.fc4 = nn.Linear(100, 50)
        self.fc5 = nn.Linear(50, 3)
    def forward(self, x):
        # x=x.to(torch.float32)
        x1 = F.relu(self.fc1(x))
        x2 = F.relu(self.fc2(x))
        x3 = torch.cat((x1, x2), dim=1)
        x = self.fc3(x3)
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x
    
class ufunc_aevs(nn.Module):
    def __init__(self):
        super(ufunc_aevs, self).__init__()
        self.fc1 = nn.Linear(64, 1000)
        self.fc2 = nn.Linear(64, 1000)
        self.fc3 = nn.Linear(2000, 1000)
        self.fc4 = nn.Linear(1000, 50)
        self.fc5 = nn.Linear(50, 3)
    def forward(self, x):
        #x=x.to(torch.float32)
        x1 = F.relu(self.fc1(x))
        x2 = F.relu(self.fc2(x))
        x3 = torch.cat((x1, x2), dim=1)
        x = self.fc3(x3)
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x
    
class ufunc_rl(nn.Module):
    def __init__(self):
        super(ufunc_rl, self).__init__()
        self.fc1 = nn.Linear(32, 100)
        self.fc2 = nn.Linear(32, 100)
        self.fc3 = nn.Linear(200, 100)
        self.fc4 = nn.Linear(100, 50)
        self.fc5 = nn.Linear(50, 3)
    def forward(self, x):
        x=x.to(torch.float32)
        x1 = F.relu(self.fc1(x))
        x2 = F.relu(self.fc2(x))
        x3 = torch.cat((x1, x2), dim=1)
        x = self.fc3(x3)
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x
    
class dNet(nn.Module):
    def __init__(self):
        super(dNet, self).__init__()
        self.fc1 = nn.Linear(35, 100)
        self.fc2 = nn.Linear(35, 100)
        self.fc3 = nn.Linear(200, 400)
        self.fc4 = nn.Linear(400, 50)
        self.fc5 = nn.Linear(50, 1)
    def forward(self, x):
#         x=x.to(torch.float32)
        x1 = F.relu(self.fc1(x))
        x2 = F.relu(self.fc2(x))
        x3 = torch.cat((x1, x2), dim=1)
        x = self.fc3(x3)
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x
    
class LyapunovNet(nn.Module):
    def __init__(
        self,
        n_states:int = 4, 
        n_hiddens:int = 128
    ):
        super(LyapunovNet, self).__init__()
        self.fc1 = nn.Linear(n_states, n_hiddens)
        self.fc2 = nn.Linear(n_hiddens, n_hiddens)
        self.fc3 = nn.Linear(n_hiddens, 2)

    def forward(self, x):
        # x = x.clone().detach().requires_grad_(True)
        s = self.fc1(x)  # -->[b, n_hiddens]
        # s = F.relu(s)
        s = torch.tanh(s)
        s = self.fc2(s)  # -->[b, n_hiddens]
        # s = F.relu(s)
        s = torch.tanh(s)
        V = self.fc3(s)  # -->[b, 1]
        # V = 0.5 * torch.pow(V,2) # semi-positive definite
        return V
    
    def V(self, x):
        return self.forward(x)[:,0]

    def V_with_JV(self, x):
        x = x.clone().detach().requires_grad_(True)
        V = self.forward(x)
        JV = torch.autograd.grad(V, x)
        return V, JV