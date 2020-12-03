""" Modify Pendulum Environment to be able to use a mlp reward """
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.envs.classic_control.pendulum import PendulumEnv
import gym.envs.classic_control.pendulum

class PendulumSSRLEnv(PendulumEnv):
    def __init__(self):
        super(PendulumSSRLEnv, self).__init__()
        self.RewardMLP = RewardNet((self.action_space.shape[-1] + self.observation_space.shape[-1])) 
        self.D_piRL = None
        self.D_samp = None
        self.T = env._max_episode_steps
        self.step_ = 0
        self.episode_train = 10
        self.state_action_record = []

    def step(self, u):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering
        costs = angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)

        newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
        newth = th + newthdot * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)

        self.state = np.array([newth, newthdot])

        # Add code to:
        #   * record state action results from the policy being learned 
        #   * add a row to D_samp after each episode 
        #   * stop and train every "episode_train" episodes 
        #   * record state 
        #   * Do a forward pass on the current state and action to get an estimated reward 
        self.state_action_record.extend((self._get_obs()[0], self._get_obs()[1], self._get_obs()[2], u)) 
        if (self.step_ % self.T) == 0:
            np.concatenate((self.D_samp, np.array(self.state_action_record)))
            self.state_action_record = []
        if (self.step_ % (self.T * self.episode_train)) == 0:
            self.train_reward()

        self.step_ += 1
        example = torch.tensor([self._get_obs()[0], self._get_obs()[1], self._get_obs()[2], u])
        reward = self.RewardMLP(example)

        return self._get_obs(), reward, False, {}
        #return self._get_obs(), -costs, False, {}

    def generate_piRL_samples(self, env_piRL, num_piRL_samples=30):
        
        self.D_piRL = np.empty([num_piRL_samples, self.T*(self.action_space.shape[-1] + self.observation_space.shape[-1])])
        for i_episode in range(num_piRL_samples):
            ep = []
            obs = env_piRL.reset()
            for t in range(self.T):
                action, _states = model_piRL.predict(obs)
                ep.append(np.concatenate((obs,action))
            
            self.D_piRL[i_episode,:] = np.array(ep).reshape(-1)

        self.D_samp = self.D_piRL.copy()

    def train_reward(self, num_train_stps=10, batch_size=4):
        
        model = self.RewardMLP
        loss_func = RewardLoss()
        loss_func.requires_grad = True
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Intialize Weights with He Initialization
        def weights_init(m):
            if (type(m) == torch.nn.Linear) 
                torch.nn.init.kaiming_uniform_(m.weight)
        weights_init(model)

        D_train = np.concatenate((self.D_piRL, self.D_samp))

        for stp in range(num_train_stps):
            i = np.random.choice(D_train.shape[0], size=args.batch_size, replace=False)
            trajectory = torch.from_numpy(D_train[i].astype(np.float32))
            
            # Forward Pass
            outputs = model(trajectory)

            # Compute Loss and train 
            loss = loss_func(self.D_piRL, self.D_samp, model)
            optimizer.zero_grad()  
            loss.backward()        
            optimizer.step()       

class RewardLoss(nn.Module):
    def __init__(self):
        super(RewardLoss, self).__init__()

    def forward(self, D_piRL, D_samp, reward_model):
        """ Calculate (negative) log-likelihood of trajectories, reward model parameters   
        D_piRL: arrary of size (N, T*V)
        D_samp: arrary of size (M, T*V)
                N = num samples from piRL, M = num samples from piTheta (and piRL),
                T = episode length, V = (size of action space) + (size of observation space)
        reward_model: mlp, current estimate of reward function 

        return: Negative of eq. 3 from Generalizing Skills paper 
        """

        # Estimate partition function, Z 
        inner_sum = 0
        for i in range(D_samp.shape[0]):
            numerator = torch.exp(reward_model(D_samp[i,:]))
            denominator = 1 / (self.max_torque - (-1.0 * self.max_torque))
            inner_sum += (numerator/denominator)
        
        log_inner_sum = torch.log(inner_sum)

        # Now estimate Likelihood
        outer_sum = 0 
        for i in range(D_piRL.shape[0]):
            outer_sum += reward_model(D_piRL[i,:]) - log_inner_sum 
        
        return -1.0 * outer_sum 

class RewardNet(nn.Module):
    def __init__(self, input_size, h1_size=30, h2_size=30):
        super(reward_net, self).__init__()

        # Inputs 
        self.input_size = input_size
        self.h1_size = h1_size
        self.h2_size = h2_size

        # Fully Connected Layers 
        self.linear1 = nn.Linear(self.input_size, self.h1_size)
        self.linear2 = nn.Linear(self.h1_size, self.h2_size)
        self.linear3 = nn.Linear(self.h2_size, 1)

        # Activations
        self.relu = nn.ReLU()
    
    def forward(self, input_data):
        out = self.relu(self.linear1(input_data))
        out = self.relu(self.linear2(out))
        return self.linear3(out)

