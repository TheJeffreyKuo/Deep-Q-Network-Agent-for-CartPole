import gymnasium as gym
import torch
import numpy as np
import os
from config import Config
from agent import Agent

def evaluate(agent, env, episodes):
    agent.q_net.eval()
    rewards = []
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            state_v = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(agent.device)
            with torch.no_grad():
                action = agent.q_net(state_v).max(1)[1].item()
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        rewards.append(total_reward)
    agent.q_net.train()
    return np.mean(rewards)

def main():
    config = Config("src/config/config_dqn_cartpole.yaml")
    train_env = gym.make(config.env_id, render_mode=None)
    state_dim = train_env.observation_space.shape[0]
    action_dim = train_env.action_space.n
    agent = Agent(config, state_dim, action_dim)
    state, _ = train_env.reset()
    episode = 1
    episode_reward = 0

    for t in range(1, config.total_timesteps + 1):
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = train_env.step(action)
        done = terminated or truncated
        agent.memory.push((state, action, reward, next_state, float(done)))
        state = next_state
        episode_reward += reward

        agent.optimize()
        if t % config.target_update == 0:
            agent.update_target()

        if done:
            print(f"Episode {episode}: Reward = {episode_reward}")
            episode += 1
            state, _ = train_env.reset()
            episode_reward = 0

    os.makedirs("models", exist_ok=True)
    torch.save(agent.q_net.state_dict(), "models/cartpole_final.pth")
    avg_score = evaluate(agent, train_env, config.eval_episodes)
    print(f"Average score over {config.eval_episodes} evaluation episodes: {avg_score:.2f}")

    if hasattr(config, 'render_final') and config.render_final:
        render_env = gym.make(config.env_id, render_mode="human")
        state, _ = render_env.reset()
        done = False
        while not done:
            render_env.render()
            state_v = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(agent.device)
            with torch.no_grad():
                action = agent.q_net(state_v).max(1)[1].item()
            state, _, terminated, truncated, _ = render_env.step(action)
            done = terminated or truncated
        render_env.close()

if __name__ == "__main__":
    main()
