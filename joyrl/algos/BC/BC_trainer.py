import numpy as np
import gym
import torch
from torch import nn
import torch.nn.functional as F
import argparse
import tqdm
import os
import matplotlib.pyplot as plt

class BC_Network(nn.Module):
    
    def __init__(self, hidden, image_shape, input_dim, num_actions, image_obs=False):
        super(BC_Network, self).__init__()
        self.num_actions = num_actions
        self.image_obs = image_obs
        layers = []
        if image_obs:
            self.conv1 = nn.Conv2d(input_dim, 32, 8, stride=4)
            self.conv2 = nn.Conv2d(32, 128, 4, stride=2)
            self.conv3 = nn.Conv2d(128, 256, 3, stride=1)

            c_out = self.conv3(self.conv2(self.conv1(torch.randn(1, input_dim, *image_shape))))
            self.conv3_size = np.prod(c_out.shape)
            self.fc1 = nn.Linear(self.conv3_size, 512)
            # self.fc1 = nn.Linear(input_dim, 512)

            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, num_actions)
        else:
            layers.append(nn.Linear(input_dim, hidden[0]))
            for i in range(len(hidden)-1):
                layers.append(nn.Linear(hidden[i], hidden[i+1]))
            layers.append(nn.Linear(hidden[-1], num_actions))
        
        self.network = nn.Sequential(*layers)

    
    
    def forward(self, x):
        return self.network(x)

class BC:
    def __init__(self, cfg, model_dir):
        self.cfg = cfg
        self.model_dir = model_dir
        self.device = cfg.device
        
        self.bc_network = BC_Network(cfg.hidden, cfg.image_shape, cfg.input_dim, cfg.n_actions).to(self.device)
        self.batch_size = cfg.batch_size

        self.num_epochs = self.cfg.train_eps
        self.optimizer = torch.optim.Adam(self.bc_network.parameters(), lr = self.cfg.lr)
        
        
    def train(self, train_dataloader):
        plt.figure()
        epoch_train_loss = []
        self.bc_network.train()
        for epoch in range(1, self.cfg.train_eps):

            temp_train_epoch = []

            with tqdm.tqdm(train_dataloader, unit='batch') as tepoch:
                
                for batch_idx, (data, label) in enumerate(tepoch):
                
                    self.optimizer.zero_grad()

                    data_tensor = data.float().clone().detach().to(self.device)
                    label_tensor = label.float().clone().detach().to(self.device)

                    network_prediction = self.bc_network(data_tensor)
                    if self.cfg.continuous_action:
                        train_loss = torch.nn.MSELoss()(network_prediction, label_tensor)
                    else:
                        train_loss = torch.nn.CrossEntropyLoss()(network_prediction, label_tensor)

                    train_loss.backward()

                    self.optimizer.step()
                    # training_losses.append(train_loss.item())
                    temp_train_epoch.append(train_loss.item())
                
                    
                    tepoch.set_description(f'Train Epoch {epoch}')
                    tepoch.set_postfix(loss = train_loss.item())

            epoch_train_loss.append(np.array(temp_train_epoch).mean())
            if epoch % 50 == 0 or epoch == self.num_epochs - 1:
                with torch.no_grad():
                    self.save_policy(epoch)
            if epoch % 1 == 0:
                self._plot(epoch_train_loss)
    
    def load_policy(self, idx):
        model_path = f'{self.model_dir}/model_{idx}.pt'
        self.bc_network.load_state_dict(torch.load(model_path))
        return self.bc_network

    def save_policy(self, idx):
        if not os.path.exists(f'{self.model_dir}'):
            os.makedirs(f'{self.model_dir}')
        save_path =  f'{self.model_dir}/model_{idx}.pt'
        torch.save(self.bc_network.state_dict(), save_path)
        print('model saved')

    def _plot(self, loss):
        if not os.path.exists(f'{self.model_dir}'):
            os.makedirs(f'{self.model_dir}')
        plt.plot(loss)
        plt.savefig(f'{self.model_dir}/loss.png')
        


