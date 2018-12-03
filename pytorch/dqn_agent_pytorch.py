import numpy as np
import random
from collections import namedtuple, deque
import torch
import torch.nn.functional as F
import torch.optim as optim


BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate
UPDATE_EVERY = 4        # how often to update the network
devc = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

E = 0.01
A = 0.6
B = 0.4


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, network, state_size, action_size, episode_max=2000, seed=42, save=True, save_loc='models/', save_every=100, checkpoint_file=None):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.args = (network, state_size, action_size,
                     seed, save, save_loc, save_every)

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        self.qnetwork_local = network(
            state_size, action_size, seed).to(devc)
        self.qnetwork_target = network(
            state_size, action_size, seed).to(devc)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        self.t_step = 0
        
        self.loss = 0
        self.learning_steps = 0

        self.eps = 0.0
        self.eps_end = 0.0
        self.eps_decay = 0.0

        self.episode_counter = 0
        self.episode_max = episode_max

    def set_epsilon(self, epsilon_start, epsilon_end, episodes_till_cap):
        self.eps = epsilon_start
        self.eps_end = epsilon_end
        self.eps_decay = epsilon_end**(1/episodes_till_cap)

    def reset_episode(self):
        self.episode_counter += 1
        self.eps = max(self.eps_end, self.eps_decay*self.eps)

    def step(self, state, action, reward, next_state, done):
        """ Stores SARS in memory for further processing and teaches agent based

        Args:
            state (array_like): state before taking action
            action (int): taken action
            reward (float): reward for taking action
            next_state (array_like): state after taking action
            done (bool): whether action ended episode
        """
        self.memory.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, mode="train"):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            mode (str): "train" or "test", for strategy choosing
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(devc)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > self.eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    # def learn(self, experiences, gamma):
    #     """Update value parameters using given batch of experience tuples.

    #     Params
    #     ======
    #         experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
    #         gamma (float): discount factor
    #     """
    #     states, actions, rewards, next_states, dones = experiences

    #     q_target_output = self.qnetwork_target.forward(next_states)
    #     Q_targets_next = np.expand_dims(
    #         np.amax(q_target_output, 1), 1)
    #     Q_targets = rewards + (gamma*Q_targets_next*(1-dones))

    #     reduced_loss, result = self.qnetwork_local.train(states, Q_targets, actions)
    #     self._update_loss(reduced_loss)

    #     self.soft_update()

    # def soft_update(self):
    #     self.sess.run(self.soft_update_op)

    def learn(self, experiences, gamma):
        '''Update value parameters using given batch of experience tuples.
        :param experiences: Tuple[torch.Tensor]. tuple of (s, a, r, s', done)
        :param gamma: float. discount factor
        '''
        states, actions, rewards, next_states, dones = experiences
        max_Qhat = self.qnetwork_target(next_states).detach().max(1)[
            0].unsqueeze(1)
        Q_target = rewards + (gamma * max_Qhat * (1 - dones))
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        loss = F.mse_loss(Q_expected, Q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        '''Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        :param local_model: PyTorch model. weights will be copied from
        :param target_model: PyTorch model. weights will be copied to
        :param tau: float. interpolation parameter
        '''
        iter_params = zip(target_model.parameters(), local_model.parameters())
        for target_param, local_param in iter_params:
            tensor_aux = tau*local_param.data + (1.0-tau)*target_param.data
            target_param.data.copy_(tensor_aux)

    def _update_loss(self, loss):
        self.loss = self.loss*self.learning_steps / \
            (self.learning_steps+1) + loss/(self.learning_steps+1)
        self.learning_steps += 1


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples.

    Attributes:
        action_size (int): dimension of each action
        buffer_size (int): maximum size of buffer
        batch_size (int): size of each training batch
        seed (int): random seed 
    """

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object."""
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=[
                                     "state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, np.clip(
            reward, -1, 1), next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences
                                             if e is not None])).float().to(devc)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences
                                              if e is not None])).long().to(devc)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences
                                              if e is not None])).float().to(devc)
        next_states = torch.from_numpy(np.vstack([e.next_state
                                                  for e in experiences
                                                  if e is not None])).float().to(devc)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences
                                            if e is not None]).astype(np.uint8)).float().to(devc)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
