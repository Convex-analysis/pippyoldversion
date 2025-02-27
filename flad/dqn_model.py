import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class QNetwork(nn.Module):
    """Neural network model for Q-function approximation."""
    
    def __init__(self, state_size, action_size, hidden_size=128, dueling=True):
        """
        Args:
            state_size: Dimension of state space
            action_size: Dimension of action space
            hidden_size: Size of hidden layers
            dueling: Whether to use dueling network architecture
        """
        super(QNetwork, self).__init__()
        
        self.dueling = dueling
        
        # Feature extraction layers
        self.features = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        
        if dueling:
            # Dueling architecture for better policy evaluation
            self.advantage_stream = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, action_size)
            )
            
            self.value_stream = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, 1)
            )
        else:
            # Standard Q-value output
            self.q_values = nn.Linear(hidden_size, action_size)
    
    def forward(self, x):
        """Forward pass through the network."""
        features = self.features(x)
        
        if self.dueling:
            advantage = self.advantage_stream(features)
            value = self.value_stream(features)
            
            # Combine value and advantage to get Q-values
            # Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
            q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        else:
            q_values = self.q_values(features)
            
        return q_values


class ReplayBuffer:
    """Standard experience replay buffer for DQN training."""
    
    def __init__(self, capacity=10000):
        """
        Args:
            capacity: Maximum size of the buffer
        """
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample a random batch from the buffer."""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self):
        """Return current size of the buffer."""
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """Prioritized experience replay buffer for DQN training."""
    
    def __init__(self, capacity=10000, alpha=0.6, beta_start=0.4, beta_increment=0.001):
        """
        Args:
            capacity: Maximum size of the buffer
            alpha: How much prioritization to use (0=none, 1=full)
            beta_start: Starting value of importance sampling correction
            beta_increment: Increment of beta per sampling
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta_start
        self.beta_increment = beta_increment
        self.buffer = deque(maxlen=capacity)
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
    
    def add(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        max_priority = self.priorities.max() if self.size > 0 else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size):
        """Sample a batch from the buffer with priorities."""
        if self.size < batch_size:
            # If not enough samples, return all available
            return random.sample(list(self.buffer), self.size), None, None
        
        # Increase beta for better importance sampling correction
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Calculate sampling probabilities from priorities
        priorities = self.priorities[:self.size]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices based on probabilities
        indices = np.random.choice(self.size, batch_size, replace=False, p=probabilities)
        
        # Calculate importance sampling weights
        weights = (self.size * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize weights
        
        samples = [self.buffer[idx] for idx in indices]
        return samples, indices, weights
    
    def update_priorities(self, indices, priorities):
        """Update priorities for sampled experiences."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
    
    def __len__(self):
        """Return current size of the buffer."""
        return self.size


class DQNPipelineModel:
    """
    DQN model for pipeline optimization in FLAD clusters.
    
    This implementation follows the SWIFT algorithm's approach for pipeline generation,
    where the DQN state space captures:
    1. Available model capacity (M_cap - sum(M^v_cap))
    2. Current model partitions (v, M^v_cap)
    3. Memory efficiency ratios (M^v_cap/mem_v) for each vehicle
    4. Computation and communication times for each vehicle
    5. The execution path derived from the cluster DAG
    
    The action space integrates:
    1. Partition assignment (which vehicle and how much model capacity)
    2. Scheduling decisions (execution sequence within DAG constraints)
    
    The reward function balances:
    1. Performance optimization (minimizing computation and communication times)
    2. Constraint satisfaction (memory limits, non-overlapping partitions, DAG precedence)
    """
    
    def __init__(self, state_size, action_size, learning_rate=0.001, discount_factor=0.95,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, batch_size=64,
                 prioritized_replay=True, double_dqn=True, dueling_network=True):
        """
        Args:
            state_size: Dimension of state space
            action_size: Dimension of action space
            learning_rate: Learning rate for optimizer
            discount_factor: Discount factor for future rewards (gamma)
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Rate of exploration decay
            batch_size: Batch size for training
            prioritized_replay: Whether to use prioritized experience replay
            double_dqn: Whether to use double DQN algorithm
            dueling_network: Whether to use dueling network architecture
        """
        self.state_size = state_size
        self.action_size = action_size
        self.discount_factor = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.prioritized_replay = prioritized_replay
        self.double_dqn = double_dqn
        
        # Initialize Q-networks
        self.q_network = QNetwork(state_size, action_size, dueling=dueling_network)
        self.target_network = QNetwork(state_size, action_size, dueling=dueling_network)
        self.update_target_network()
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Initialize replay buffer
        if prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer()
        else:
            self.replay_buffer = ReplayBuffer()
        
        # Track training metrics
        self.loss_history = []
        self.reward_history = []
        self.training_steps = 0
        
        # Set device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.q_network.to(self.device)
        self.target_network.to(self.device)
    
    def update_target_network(self):
        """Update the target network with current q_network weights."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def get_action(self, state, explore=True):
        """
        Select action using epsilon-greedy policy with constraint awareness.
        
        Args:
            state: Current state
            explore: Whether to use exploration (True) or exploitation only (False)
            
        Returns:
            Selected action index
        """
        if explore and random.random() < self.epsilon:
            # Exploration: random action
            return random.randrange(self.action_size)
        else:
            # Exploitation: best action from Q-network
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return q_values.cpu().data.numpy().argmax()
    
    def train(self, state, action, reward, next_state, done):
        """
        Add experience to buffer and perform training step.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        # Add experience to replay buffer
        self.replay_buffer.add(state, action, reward, next_state, done)
        
        # Train if enough samples in buffer
        if len(self.replay_buffer) >= self.batch_size:
            self._train_step()
            self.training_steps += 1
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            
        # Record reward for tracking
        self.reward_history.append(reward)
    
    def _train_step(self):
        """Perform one training step using a batch from replay buffer."""
        # Sample batch from replay buffer
        if self.prioritized_replay:
            batch, indices, weights = self.replay_buffer.sample(self.batch_size)
            if weights is not None:
                weights = torch.FloatTensor(weights).to(self.device)
        else:
            batch = self.replay_buffer.sample(self.batch_size)
            indices, weights = None, None
        
        # Prepare batch data - using numpy.array() for efficient tensor conversion
        states = torch.FloatTensor(np.array([exp[0] for exp in batch])).to(self.device)
        actions = torch.LongTensor(np.array([[exp[1]] for exp in batch])).to(self.device)
        rewards = torch.FloatTensor(np.array([[exp[2]] for exp in batch])).to(self.device)
        next_states = torch.FloatTensor(np.array([exp[3] for exp in batch])).to(self.device)
        dones = torch.FloatTensor(np.array([[int(exp[4])] for exp in batch])).to(self.device)
        
        # Get current Q-values
        current_q_values = self.q_network(states).gather(1, actions)
        
        # Calculate target Q-values using Double DQN or regular DQN
        with torch.no_grad():
            if self.double_dqn:
                # Double DQN: Get actions from current network
                next_actions = self.q_network(next_states).max(1)[1].unsqueeze(1)
                # Get Q-values from target network
                max_next_q_values = self.target_network(next_states).gather(1, next_actions)
            else:
                # Regular DQN: Get max Q-values directly from target network
                max_next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
            
            # Compute TD targets
            target_q_values = rewards + (1 - dones) * self.discount_factor * max_next_q_values
        
        # Compute loss with importance sampling weights if using prioritized replay
        td_errors = target_q_values - current_q_values
        if weights is not None:
            # Prioritized replay: weight the MSE loss by importance sampling weights
            loss = (weights * (td_errors ** 2)).mean()
        else:
            loss = nn.MSELoss()(current_q_values, target_q_values)
            
        self.loss_history.append(loss.item())
        
        # Update priorities if using prioritized replay
        if self.prioritized_replay and indices is not None:
            # Update priorities with TD errors
            new_priorities = (td_errors.abs().detach().cpu().numpy() + 1e-6).flatten()
            self.replay_buffer.update_priorities(indices, new_priorities)
        
        # Update network
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
    
    def update_target_if_needed(self, update_interval=100):
        """
        Update target network if needed based on update interval.
        
        Args:
            update_interval: Number of training steps between target updates
        """
        if self.training_steps > 0 and self.training_steps % update_interval == 0:
            self.update_target_network()
            return True
        return False
    
    def save_model(self, path):
        """Save model to file."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'loss_history': self.loss_history,
            'reward_history': self.reward_history,
            'training_steps': self.training_steps
        }, path)
        print(f"Model successfully saved to {path}")
    
    def load_model(self, path):
        """Load model from file."""
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        
        # Move loaded models to the proper device
        self.q_network.to(self.device)
        self.target_network.to(self.device)
        
        # Load additional state if saved
        if 'loss_history' in checkpoint:
            self.loss_history = checkpoint['loss_history']
        if 'reward_history' in checkpoint:
            self.reward_history = checkpoint['reward_history']
        if 'training_steps' in checkpoint:
            self.training_steps = checkpoint['training_steps']
        
        print(f"Model successfully loaded from {path}")
        
    def train_from_experiences(self, experiences, epochs=1):
        """
        Train the model from a list of experiences without adding to replay buffer.
        
        Args:
            experiences: List of (state, action, reward, next_state, done) tuples
            epochs: Number of training epochs to perform
        """
        # Add experiences to buffer without training
        for state, action, reward, next_state, done in experiences:
            self.replay_buffer.add(state, action, reward, next_state, done)
            
        # Train for specified number of epochs
        for _ in range(epochs):
            if len(self.replay_buffer) >= self.batch_size:
                self._train_step()
                self.training_steps += 1
                
        # Update target network after training
        self.update_target_network()
        
    def get_q_values(self, state):
        """
        Get Q-values for all actions in the given state.
        
        Args:
            state: Current state
            
        Returns:
            Array of Q-values for each action
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.cpu().data.numpy()[0]