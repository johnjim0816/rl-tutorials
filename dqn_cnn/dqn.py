import random
import math
import torch
import torch.optim as optim
import torch.nn.functional as F
from memory import ReplayBuffer
from model import CNN


class DQN:
    def __init__(self, screen_height=0, screen_width=0, n_actions=0, gamma=0.999, epsilon_start=0.9, epsilon_end=0.05, epsilon_decay=200, memory_capacity=10000, batch_size=128, device="cpu"):
        self.actions_count = 0 
        self.n_actions = n_actions  # 总的动作个数
        self.device = device  # 设备，cpu或gpu等
        self.gamma = gamma
        # e-greedy策略相关参数
        self.epsilon = 0
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.policy_net = CNN(screen_height, screen_width,
                              n_actions).to(self.device)
        self.target_net = CNN(screen_height, screen_width,
                              n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict()) # target_net的初始模型参数完全复制policy_net
        self.target_net.eval()  # 不启用 BatchNormalization 和 Dropout
        self.optimizer = optim.RMSprop(self.policy_net.parameters()) # 可查parameters()与state_dict()的区别，前者require_grad=True
        self.loss = 0
        self.memory = ReplayBuffer(memory_capacity)
        

    def select_action(self, state):
        '''选择动作
        Args:
            state [array]: [description]
        Returns:
            action [array]: [description]
        '''
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * self.actions_count / self.epsilon_decay)
        self.actions_count += 1
        if random.random() > self.epsilon:   
            with torch.no_grad():
                q_value = self.policy_net(state) # q_value比如tensor([[-0.2522,  0.3887]])
                # tensor.max(1)返回每行的最大值以及对应的下标，
                # 如torch.return_types.max(values=tensor([10.3587]),indices=tensor([0]))
                # 所以tensor.max(1)[1]返回最大值对应的下标，即action
                action = q_value.max(1)[1].view(1, 1)  # 注意这里action是个张量，如tensor([1])
                return action
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = self.memory.Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)   
        action_batch = torch.cat(batch.action) 
        reward_batch = torch.cat(batch.reward) # tensor([1., 1.,...,])
        

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(
            state_batch).gather(1, action_batch) #tensor([[ 1.1217],...,[ 0.8314]])

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=self.device)

        next_state_values[non_final_mask] = self.target_net(
            non_final_next_states).max(1)[0].detach()

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch # tensor([0.9685, 0.9683,...,])
        
        # Compute Huber loss
        self.loss = F.smooth_l1_loss(
            state_action_values, expected_state_action_values.unsqueeze(1))  # .unsqueeze增加一个维度
        # Optimize the model
        self.optimizer.zero_grad() # zero_grad clears old gradients from the last step (otherwise you’d just accumulate the gradients from all loss.backward() calls).
        self.loss.backward() # loss.backward() computes the derivative of the loss w.r.t. the parameters (or anything requiring gradients) using backpropagation.
        for param in self.policy_net.parameters(): # clip防止梯度爆炸
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step() # causes the optimizer to take a step based on the gradients of the parameters.


if __name__ == "__main__":
    dqn = DQN()
