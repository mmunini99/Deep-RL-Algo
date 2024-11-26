import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
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


# NAF
class NAF_DQN(nn.Module):

    def __init__(self, dim_action, max_action, device):
        super(NAF_DQN, self).__init__()
        self.device = device
        self.max_action = max_action.to(self.device)
        self.dim = dim_action
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=4, padding=2)  
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=2, padding=2)  
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1)  
        # Batch Normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)

        # Fully connected layers
        self.fc1 = nn.Linear(12544, 2048)  # Flattened size after convolution layers
        self.fc2 = nn.Linear(2048, 512)

        self.mu = nn.Linear(512, dim_action) #action
        self.value = nn.Linear(512, 1) #value function
        self.matrix = nn.Linear(512, int(dim_action * (dim_action + 1) / 2)) #only lower triangle is needed for P matrix. Values of advantages.

    def forward(self, x, action):
        # Convolutional block with ReLU and Batch Normalization
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Flattening the output for the fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layers with ReLU
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Output layer
        mu = torch.tanh(self.mu(x)) * self.max_action
        V = self.value(x)
        P = torch.tanh(self.matrix(x)) #scaling to avoid explosion of gradients


        L = torch.zeros((x.shape[0], self.dim, self.dim)).to(self.device) # batch_size x action dims x action dims --> bathc size is the state
        tril_indices = torch.tril_indices(row=self.dim, col=self.dim, offset=0).to(self.device) # define the valid indexes to fill a triangualr matrix

        L[:, tril_indices[0], tril_indices[1]] = P
        L.diagonal(dim1=1,dim2=2).exp_() # to ensure that P is a pos. definite matrix
        P_complete = L * L.transpose(2, 1) #

        u_mu = (action-mu).unsqueeze(dim=1)
        u_mu_t = u_mu.transpose(1, 2)

        adv = - 1/2 * u_mu @ P_complete @ u_mu_t
        adv = adv.squeeze(dim=-1)
        return V + adv


    @torch.no_grad() #selection of action don't partecipate in the learning part.
    def mean_action(self, x): #compute action highest Q-value
        # Convolutional block with ReLU and Batch Normalization
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Flattening the output for the fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layers with ReLU
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.mu(x)
        x = torch.tanh(x) * self.max_action #map from [-inf, +inf] to [-1,1] and then this is scaled by multiplting with the max action. --> non mi piace troppo se non è simmetrico.
        return x

    @torch.no_grad()
    def value_state(self, x):
        # Convolutional block with ReLU and Batch Normalization
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Flattening the output for the fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layers with ReLU
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.value(x)
        return x



# TD3
class POLICY__TD3(nn.Module):

    def __init__(self, dim_action, max_value, min_value, device):
        super(POLICY__TD3, self).__init__()
        self.device = device
        self.max_value = max_value.to(self.device)
        self.min_value = min_value.to(self.device)
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=4, padding=2)  
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=2, padding=2) 
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1) 

        # Batch Normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)

        # Fully connected layers
        self.fc1 = nn.Linear(12544, 2048)  # Flattened size after convolution layers
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, dim_action)


    def forward(self, x, sd_value, noise_clamp=None):

        x = x.clone()
        # Convolutional block with ReLU and Batch Normalization
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Flattening the output for the fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layers with ReLU
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.tanh(self.fc3(x))

        # Output layer
        mean_value = x*self.max_value
        noise = torch.normal(0, sd_value, mean_value.size())
        if noise_clamp is not None:
            noise = torch.clamp(noise, -noise_clamp, noise_clamp)
        noise = noise.to(self.device)
        mean_value = mean_value + noise
        return torch.max(torch.min(mean_value, self.max_value), self.min_value)


    def mean_action_single(self, x):

        x = x.clone()
        # Convolutional block with ReLU and Batch Normalization
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Flattening the output for the fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layers with ReLU
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.tanh(self.fc3(x))

        # Output layer
        return x*self.max_value






class Q__TD3(nn.Module):

    def __init__(self, action_dim, device):
        super(Q__TD3, self).__init__()
        self.device = device
        self.action_dim = action_dim
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=4, padding=2)  
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=2, padding=2) 
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1) 

        # Batch Normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)

        # Fully connected layers
        self.fc1 = nn.Linear(12544+self.action_dim, 2048)  # Flattened size after convolution layers
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 1)

    def forward(self, x, action_array):

        # Convolutional block with ReLU and Batch Normalization
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))




        # Flattening the output for the fully connected layers
        x = x.view(x.size(0), -1)

        act = torch.tensor(action_array)

        act = act.to(self.device)


        # Include the action
        x = torch.cat([x, act], dim=-1)




        # Fully connected layers with ReLU
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Output layer
        x = self.fc3(x)

        return x



# PPO
class POLICY__PPO(nn.Module):

    def __init__(self, dim_action, max_value):
        super(POLICY__PPO, self).__init__()
        self.max_value = max_value
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=4, padding=2) 
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=2, padding=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1)

        # Batch Normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)

        # Fully connected layers
        self.fc1 = nn.Linear(12544, 2048)  # Flattened size after convolution layers
        self.fc2 = nn.Linear(2048, 512)
        self.mean = nn.Linear(512, dim_action)
        self.sd = nn.Linear(512, dim_action)

    def forward(self, x):

        x = x.clone()
        # Convolutional block with ReLU and Batch Normalization
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Flattening the output for the fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layers with ReLU
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Output layer
        mu_param = self.mean(x)
        sd_param = self.sd(x)
        sd_param = F.softplus(sd_param) + 0.0001 #to ensure sd always > 0

        pdf = Normal(mu_param, sd_param)
        action_sample = pdf.sample()
        log_probability = pdf.log_prob(action_sample)
        log_prob_single = log_probability.sum(dim=-1, keepdim = True) #to convert log prob for each action in a single log prob.
        log_prob_single -= (2*(np.log(2)-action_sample-F.softplus(-2*action_sample))).sum(dim=-1, keepdim = True) #stability of log prob
        action = torch.tanh(action_sample)*self.max_value

        return action, log_prob_single

    def log_prob_and_entropy(self, x, action_array):

        x = x.clone()
        # Convolutional block with ReLU and Batch Normalization
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Flattening the output for the fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layers with ReLU
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Output layer
        mu_param = self.mean(x)
        sd_param = self.sd(x)
        sd_param = F.softplus(sd_param) + 0.0001 #to ensure sd always > 0

        pdf = Normal(mu_param, sd_param)
        log_probability = pdf.log_prob(action_array)
        log_prob_single = log_probability.sum(dim=-1, keepdim = True) #to convert log prob for each action in a single log prob.
        log_prob_single -= (2*(np.log(2)-action_array-F.softplus(-2*action_array))).sum(dim=-1, keepdim = True) #stability of log prob
        entropy = pdf.entropy().mean()

        return entropy, log_prob_single.squeeze(1)



class VALUE__PPO(nn.Module):

    def __init__(self):
        super(VALUE__PPO, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=4, padding=2)  
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=2, padding=2) 
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1) 

        # Batch Normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)

        # Fully connected layers
        self.fc1 = nn.Linear(12544, 2048)  # Flattened size after convolution layers
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 1)

    def forward(self, x):

        # Convolutional block with ReLU and Batch Normalization
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))




        # Flattening the output for the fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layers with ReLU
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Output layer
        x = self.fc3(x)

        return x


# SAC
class POLICY__SAC(nn.Module):

    def __init__(self, dim_action, max_value):
        super(POLICY__SAC, self).__init__()
        self.max_value = max_value
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=4, padding=2)  
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=2, padding=2) 
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1) 

        # Batch Normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)

        # Fully connected layers
        self.fc1 = nn.Linear(12544, 2048)  # Flattened size after convolution layers
        self.fc2 = nn.Linear(2048, 512)
        self.mean = nn.Linear(512, dim_action)
        self.sd = nn.Linear(512, dim_action)

    def forward(self, x):

        x = x.clone()
        # Convolutional block with ReLU and Batch Normalization
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Flattening the output for the fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layers with ReLU
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Output layer
        mu_param = self.mean(x)
        sd_param = self.sd(x)
        sd_param = F.softplus(sd_param) + 0.0001 #to ensure sd always > 0

        pdf = Normal(mu_param, sd_param)
        action_sample = pdf.sample()
        log_probability = pdf.log_prob(action_sample)
        log_prob_single = log_probability.sum(dim=-1, keepdim = True) #to convert log prob for each action in a single log prob.
        log_prob_single -= (2*(np.log(2)-action_sample-F.softplus(-2*action_sample))).sum(dim=-1, keepdim = True) #stability of log prob
        action = torch.tanh(action_sample)*self.max_value

        return action, log_prob_single



class Q__SAC(nn.Module):

    def __init__(self, dim_action, device):
        super(Q__SAC, self).__init__()
        self.device = device
        self.dim_action = dim_action
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=4, padding=2)  
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=2, padding=2) 
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1) 

        # Batch Normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)

        # Fully connected layers
        self.fc1 = nn.Linear(12544+self.dim_action, 2048)  # Flattened size after convolution layers
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 1)

    def forward(self, x, action_array):

        # Convolutional block with ReLU and Batch Normalization
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))




        # Flattening the output for the fully connected layers
        x = x.view(x.size(0), -1)

        act = torch.tensor(action_array)

        act = act.to(self.device)


        # Include the action
        x = torch.cat([x, act], dim=-1)




        # Fully connected layers with ReLU
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Output layer
        x = self.fc3(x)

        return x