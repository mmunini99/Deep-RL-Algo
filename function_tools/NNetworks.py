import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# DQN
class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
# Quantile version of DQN
class QR_DQN(nn.Module):

    def __init__(self, n_observations, n_actions, n_quantiles):
        super(QR_DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions*n_quantiles)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
# IQN
class EmbeddingNN(nn.Module):

    def __init__(self, hidd_dim1, hidd_dim2, device):
        self.device = device
        super(EmbeddingNN, self).__init__()
        self.hidd_dim1 = hidd_dim1
        self.hidd_dim2 = hidd_dim2
        self.layer_cos1 = nn.Linear(hidd_dim1, hidd_dim2)


    def forward(self):
        tau = torch.rand(1)
        cosine = torch.cos((np.pi*torch.arange(1,self.hidd_dim1+1, dtype=torch.float).reshape(-1, 1))*tau)
        cosine = cosine.to(self.device)
        embedding_feature = F.relu(self.layer_cos1(cosine.transpose(0,1)))
        embedding_feature = embedding_feature.to(self.device)
        return embedding_feature
    
class IQN(nn.Module):

    def __init__(self, n_observations, n_actions, device):
        self.device = device
        super(IQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.embedding_NN = EmbeddingNN(64,128, self.device).to(self.device)
        self.layer3 = nn.Linear(128, n_actions)




    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.embedding_NN()*x
        return self.layer3(x)