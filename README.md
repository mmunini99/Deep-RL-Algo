In this repo I upload some popular Deep RL algorithms :
DISCRETE ACTION
* DQN : In this implementation, it works only with MLP network and based on a like-gym environment. However, to replace the MLP with a CNN is very easy to do and in the Utilities.py and NNetworks.py scripts there is everything you need.
* QR DQN : Same as DQN. It uses a quantile version of Huber loss that can be found in Utilities.py. For changing the MLP to CNN remember to insert in the last fully-connected layer NUMBER OF ACTIONS * NUMBER OF QUANTILES as output dimensions. 
* IQN : In this implementations the n_sub_agents refers the number of parallel training in the optimization steps to do. The IQN netowrk in NNetworks.py uses an Embedding NN. Note: at the moment this class has to be improve in terms of efficiency.
CONTINUOUS ACTION
* NAF : In this implementation the neural network is based on CNN layer.
* TD3 : In this implementation the neural networks for both Policy and Q-value are based on CNN layer. Also, the Policy and Q-value are computed using two separated neural networks.
* PPO : In this implementation the neural network is based on CNN layer. However, to make it more efficient it need to be parallelized the step where the dataset of multiple episode is created.




parlare di esempi di come si usano e dell'ambiente
