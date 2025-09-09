# Project Overview
A minimal Deep Q-Network (DQN) implementation for solving the CartPole-v1 environment using PyTorch and Gymnasium. This project trains a DQN agent with experience replay and a target network to balance the pole in the CartPole-v1 environment. The agent uses epsilon-greedy exploration with decay, and a simple feedforward neural network architecture.

## Usage

1. Install dependencies:
   ```
   pip install torch gymnasium pyyaml numpy
   ```

2. Run training:
   ```
   python main.py
   ```

3. After training, a model checkpoint will be saved to `models/cartpole_final.pth`.

## Configuration Highlights

- `env_id`: Gym environment ID (default: CartPole-v1)
- `gamma`: Discount factor
- `epsilon_start`, `epsilon_end`, `epsilon_decay`: Epsilon-greedy parameters
- `learning_rate`: Adam optimizer learning rate
- `batch_size`, `buffer_size`: Replay memory parameters
- `target_update`: Frequency of target network updates
- `hidden_dim`: Neural network hidden layer size
- `total_timesteps`: Number of training steps
- `eval_episodes`: Number of evaluation episodes
- `render_final`: Whether to display the environment render after training
