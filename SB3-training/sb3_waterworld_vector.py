"""Uses Stable-Baselines3 to train agents to play the Waterworld environment using SuperSuit vector envs.

For more information, see https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html

Author: Elliot (https://github.com/elliottower)
"""
from __future__ import annotations

import glob
import os
import time
import csv

import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from scipy import stats

import gymnasium as gym 

from pettingzoo.sisl import waterworld_v4, waterworld_model1

import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.evaluation import evaluate_policy
from pettingzoo.utils.conversions import aec_to_parallel
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement

import numpy as np 
import pandas as pd

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor


def moving_average(values, window):
        """
        Smooth values by doing a moving average
        :param values: (numpy array)
        :param window: (int)
        :return: (numpy array)
        """
        weights = np.repeat(1.0, window) / window
        return np.convolve(values, weights, "valid")


def plot_results_custom(log_folder, title="Learning Curve"):
        """
        plot the results

        :param log_folder: (str) the save location of the results to plot
        :param title: (str) the title of the task to plot
        """
        x, y = ts2xy(load_results(log_folder), "timesteps")
        y = moving_average(y, window=50)
        # Truncate x
        x = x[len(x) - len(y) :]

        fig = plt.figure(title)
        plt.plot(x, y)
        plt.xlabel("Number of Timesteps")
        plt.ylabel("Rewards")
        plt.title(title + " Smoothed")
        plt.show()

class PlottingCallbackMetrics_MultiAgentMonitor(BaseCallback):
    def __init__(self, log_dir, verbose=1):
        super(PlottingCallbackMetrics, self).__init__(verbose)
        self.log_dir = log_dir

    def _on_step(self) -> bool:
        return True

    def _plot_metric(self, data, metric, agent):
        plt.figure(figsize=(10, 5))
        plt.plot(data['t'], data[f'{agent}_{metric}'])
        plt.title(f'{agent} {metric} over time')
        plt.xlabel('Timesteps')
        plt.ylabel(metric.capitalize())
        plt.savefig(f'{self.log_dir}/{agent}_{metric}_plot.png')
        plt.close()

    def on_training_end(self):
        # Load the CSV file
        df = pd.read_csv(f'{self.log_dir}/monitor.csv', skiprows=1)  # Skip the first row which contains metadata

        agents = ['agent_0', 'agent_1']  # Adjust if you have different agent names
        metrics = ['arousal', 'satiety', 'social_touch']

        # Plot individual metrics for each agent
        for agent in agents:
            for metric in metrics:
                self._plot_metric(df, metric, agent)

            # Plot combined metrics for each agent
            plt.figure(figsize=(12, 6))
            for metric in metrics:
                plt.plot(df['t'], df[f'{agent}_{metric}'], label=metric)
            plt.title(f'{agent} metrics over time')
            plt.xlabel('Timesteps')
            plt.ylabel('Value')
            plt.legend()
            plt.savefig(f'{self.log_dir}/{agent}_combined_metrics.png')
            plt.close()

        # Plot reward (which is common for both agents)
        plt.figure(figsize=(10, 5))
        plt.plot(df['t'], df['r'])
        plt.title('Reward over time')
        plt.xlabel('Timesteps')
        plt.ylabel('Reward')
        plt.savefig(f'{self.log_dir}/reward_plot.png')
        plt.close()

        # Plot all metrics for both agents in one graph
        plt.figure(figsize=(15, 8))
        for agent in agents:
            for metric in metrics:
                plt.plot(df['t'], df[f'{agent}_{metric}'], label=f'{agent} {metric}')
        plt.plot(df['t'], df['r'], label='Reward', linewidth=2)
        plt.title('All metrics over time')
        plt.xlabel('Timesteps')
        plt.ylabel('Value')
        plt.legend()
        plt.savefig(f'{self.log_dir}/all_metrics_plot.png')
        plt.close()

        print(f"Plots saved in {self.log_dir}")
    
class PlottingCallbackMetrics(BaseCallback):
    def __init__(self, log_dir, verbose=1):
        super(PlottingCallbackMetrics, self).__init__(verbose)
        self.log_dir = log_dir

    def _on_step(self) -> bool:
        return True

    def _plot_metric(self, data, metric, color, agent):
        plt.figure(figsize=(10, 5))
        plt.plot(data['t'], data[f'{agent}_{metric}'], color=color)
        plt.title(f'{agent.capitalize()} {metric.capitalize()} over time')
        plt.xlabel('Timesteps')
        plt.ylabel(f'{agent.capitalize()} {metric.capitalize()}')
        plt.savefig(f'{self.log_dir}/{agent}_{metric}_plot.png')
        plt.close()

    def on_training_end(self):
        # Load the CSV file
        df = pd.read_csv(f'{self.log_dir}/monitor.csv', skiprows=1)  # Skip the first row which contains metadata

        agents = ['agent_0', 'agent_1']
        metrics = ['r', 'arousal', 'satiety', 'social_touch']
        colors = {'r': 'b', 'arousal': 'r', 'satiety': 'g', 'social_touch': 'purple'}

        for agent in agents:
            # Plot individual metrics for each agent
            for metric in metrics:
                self._plot_metric(df, metric, colors[metric], agent)

            # Create a combined plot for each agent
            plt.figure(figsize=(12, 8))
            for metric in metrics:
                plt.plot(df['t'], df[f'{agent}_{metric}'], label=metric.capitalize(), color=colors[metric])
            plt.title(f'{agent.capitalize()} Metrics over time')
            plt.xlabel('Timesteps')
            plt.ylabel('Value')
            plt.legend()
            plt.savefig(f'{self.log_dir}/{agent}_combined_metrics_plot.png')
            plt.close()

        # Create a combined plot for both agents
        plt.figure(figsize=(15, 10))
        for agent in agents:
            for metric in metrics:
                plt.plot(df['t'], df[f'{agent}_{metric}'], label=f'{agent} {metric}', alpha=0.7)
        plt.title('All Metrics for Both Agents over time')
        plt.xlabel('Timesteps')
        plt.ylabel('Value')
        plt.legend()
        plt.savefig(f'{self.log_dir}/all_agents_combined_metrics_plot.png')
        plt.close()

        print(f"Plots saved in {self.log_dir}")

class MultiAgentMonitor(Monitor):
    def __init__(self, env, filename, info_keywords=()):
        super().__init__(env, filename, info_keywords=info_keywords)
        # Instead of using env.agents, we'll determine the number of agents from the action space
        self.num_agents = env.action_space.shape[0]  # Assuming the action space is structured for multiple agents

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.rewards.append(sum(reward))  # Sum rewards across all agents
        if done.all():  # Assuming done is a boolean array for all agents
            ep_rew = sum(self.rewards)
            ep_len = len(self.rewards)
            ep_info = {"r": ep_rew, "l": ep_len, "t": round(time.time() - self.t_start, 6)}
            for i in range(self.num_agents):
                for k in self.info_keywords:
                    ep_info[f"agent_{i}_{k}"] = info[k][i]  # Assuming info is structured as {key: [agent1_value, agent2_value, ...]}
            self.episode_returns.append(ep_rew)
            self.episode_lengths.append(ep_len)
            self.episode_times.append(time.time() - self.t_start)
            ep_info.update(self.current_reset_info)
            if self.results_writer:
                self.results_writer.write_row(ep_info)
            self.reset_info()
            self.rewards = []
        return observation, reward, done, info
    

def plot_results(log_folder, title="Learning Curve"):
    episode_rewards = np.load(os.path.join(log_folder, "episode_rewards.npy"))
    episode_lengths = np.load(os.path.join(log_folder, "episode_lengths.npy"))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # Plot episode rewards
    ax1.plot(episode_rewards)
    ax1.set_ylabel("Episode Reward")
    ax1.set_title(title + " - Episode Rewards")

    # Plot moving average of rewards
    window = min(100, len(episode_rewards))
    moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
    ax1.plot(moving_avg, color='red')
    
    # Plot episode lengths
    ax2.plot(episode_lengths)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Episode Length")
    ax2.set_title("Episode Lengths")

    plt.tight_layout()
    plt.show()


def plot_episode_rewards(ax, rewards, episode_num):
    ax.plot(rewards)
    ax.set_title(f"Rewards Over Time - Episode {episode_num}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Average Reward")
    ax.grid(True)


def train_butterfly_supersuit(
    env_fn, steps: int = 10_000, seed: int | None = 0, **env_kwargs
):

    def create_env(num_envs=8, for_eval=False):
        env = env_fn.parallel_env(**env_kwargs)
        env.reset(seed= seed + (1000 if for_eval else 0))  # Different seed for eval
        env = ss.pettingzoo_env_to_vec_env_v1(env)
        env = ss.concat_vec_envs_v1(env, num_envs, num_cpus=2, base_class="stable_baselines3")
        return env

    def create_env_multiMonitor(num_envs=8, log_dir="./logs", for_eval=False):
        env_counter = 0
        def make_env():
            nonlocal env_counter
            env = env_fn.parallel_env(**env_kwargs)
            env.reset(seed=seed + (1000 if for_eval else 0))
            env = ss.pettingzoo_env_to_vec_env_v1(env)
            # Use the custom MultiAgentMonitor with a counter for unique env ids
            env = MultiAgentMonitor(env, f"{log_dir}/env_{env_counter}", info_keywords=('arousal', 'satiety', 'social_touch'))
            env_counter += 1
            return env

        return DummyVecEnv([make_env for _ in range(num_envs)])
    
    # Set up logging directory
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    log_dir = f"./logs/{env_fn.__name__}_{time.strftime('%Y%m%d-%H%M%S')}"
    os.makedirs(log_dir, exist_ok=True)

    # Create training environment
    env = create_env(num_envs=8)
    
    # Wrap the vectorized environment with VecMonitor
    #env = VecMonitor(train_env, log_dir, info_keywords = ['arousal', 'satiety', 'social_touch'])

    print(f"Starting training on {str(env.unwrapped.metadata['name'])}.")

    # Create evaluation environment
    eval_env = create_env(num_envs=1, for_eval=True)


    def linear_schedule(initial_value: float, final_value: float, total_timesteps: int):
        """
        Linear learning rate schedule.
        
        :param initial_value: Initial learning rate.
        :param final_value: Final learning rate.
        :param total_timesteps: Total number of timesteps.
        :return: schedule that computes current learning rate depending on remaining timesteps
        """
        def func(progress_remaining: float) -> float:
            """
            Progress will decrease from 1 (beginning) to 0.
            
            :param progress_remaining:
            :return: current learning rate
            """
            return final_value + progress_remaining * (initial_value - final_value)
        
        return func


    total_timesteps = 500000  # 5 million timesteps
    eval_freq = 100_000  # Evaluate every 100,000 steps

    # Create the learning rate schedule
    learning_rate = linear_schedule(initial_value=3e-4, final_value=1e-5, total_timesteps=total_timesteps)

    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path=log_dir,
        log_path=log_dir, 
        eval_freq=eval_freq,
        n_eval_episodes=10,
        deterministic=True, 
        render=False,
        verbose=1
    )
    
    # Create the early stopping callback
    stop_train_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=5,
        min_evals=20,
        verbose=1
    )

    # Combine EvalCallback and StopTrainingOnNoModelImprovement
    callback_on_best = StopTrainingOnNoModelImprovement(max_no_improvement_evals=5, min_evals=20, verbose=1)
    eval_callback = EvalCallback(eval_env, callback_on_new_best=callback_on_best, eval_freq=eval_freq,
                                best_model_save_path=log_dir, deterministic=True, render=False)

    # Your plotting callback
    plotting_callback = PlottingCallbackMetrics(log_dir)


    # Combine all callbacks
    callbacks = [eval_callback, plotting_callback]


    # Create and train the model with the learning rate schedule
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=learning_rate)
    model.learn(total_timesteps=total_timesteps, callback=callbacks)

    print("Model has been saved.")
    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")
    env.close()
    eval_env.close()

    return log_dir

def plot_with_confidence_interval(ax, data, label, color):
    mean = np.mean(data, axis=0)
    std_error = stats.sem(data, axis=0)
    ci = 1.96 * std_error  # 95% confidence interval

    x = range(len(mean))
    ax.plot(x, mean, label=label, color=color)
    ax.fill_between(x, mean - ci, mean + ci, color=color, alpha=0.3)


def eval(env_fn, num_games: int = 100, render_mode: str | None = None, **env_kwargs):
    # Evaluate a trained agent vs a random agent
    env = env_fn.env(render_mode=render_mode, **env_kwargs)

    print(
        f"\nStarting evaluation on {env.unwrapped.metadata.get('name')} {os.path.getctime}  (num_games={num_games}, render_mode={render_mode})"
    )

    try:
        # Look for the best model in the logs directory
        latest_policy = max(
            glob.glob(f"{env.metadata['name']}*.zip"), key=os.path.getctime
        )

        print(f"Loading policy from: {latest_policy}")
    except ValueError:
        print("Policy not found.")
        exit(0)

    model = PPO.load(latest_policy)

    print(f"Starting eval on {str(env.metadata['name'])}.")
    rewards = {agent: 0 for agent in env.possible_agents}

    # Note: We train using the Parallel API but evaluate using the AEC API
    # SB3 models are designed for single-agent settings, we get around this by using he same model for every agent
    for i in range(num_games):
        env.reset(seed=i)
        print(str(i) + " game")
        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()

            for a in env.agents:
                rewards[a] += env.rewards[a]
                print("reward for agent " + str(a) + " = " + str(rewards[a]))
            if termination or truncation:
                break
            else:
                act = model.predict(obs, deterministic=True)[0]

            env.step(act)
    env.close()

    avg_reward = sum(rewards.values()) / len(rewards.values())
    print("Rewards: ", rewards)
    print(f"Avg reward: {avg_reward}")
    return avg_reward

def find_latest_policy(env_name, base_dir="."):
    log_dir = os.path.join(base_dir, "logs")
    
    # Find all directories matching the pattern
    pattern = os.path.join(log_dir, f"{env_name}*")
    matching_dirs = glob.glob(pattern)
    
    if not matching_dirs:
        print(f"No matching directories found in {log_dir}")
        return None
    
    # Find the most recent directory
    latest_dir = max(matching_dirs, key=os.path.getctime)
    
    # Look for best_model.zip in the latest directory
    best_model_path = os.path.join(latest_dir, "best_model.zip")
    
    if os.path.exists(best_model_path):
        return best_model_path
    else:
        print(f"best_model.zip not found in {latest_dir}")
        return None
    
def eval_new(seed_start, env_fn, num_games: int = 100, render_mode: str | None = None, deterministic: bool = True, save_path: str = "./eval_results", **env_kwargs):
    env = env_fn.env(render_mode=render_mode, **env_kwargs)
                                        
    env_name = env.metadata['name'] # Adjust this to match your folder naming convention
    latest_policy = find_latest_policy(env_name)
    print(latest_policy)
    model = PPO.load(latest_policy)

    
    print(f"\nStarting evaluation on {env.unwrapped.metadata.get('name')} (num_games={num_games}, render_mode={render_mode})")

    n_agents = len(env.possible_agents)
    episode_rewards = []
    episode_lengths = []
    all_agent_rewards = {agent: [] for agent in env.possible_agents}
    per_step_rewards = [] 
    
    # Ensure the save directory exists
    os.makedirs(save_path, exist_ok=True)
    csv_path = os.path.join(save_path, f"eval_results_{env.unwrapped.metadata.get('name')}_{num_games}_games_evalround_{seed_start}.csv")
    
    with open(csv_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write the header
        header = ['Episode', 'Average_Reward', 'Episode_Length'] + [f'{agent}_Reward' for agent in env.possible_agents]
        csvwriter.writerow(header)
        
        for i in range(num_games):
            env.reset(seed=seed_start*10 + i)
            print(f"Game {i + 1}")
            
            current_rewards = {agent: 0 for agent in env.possible_agents}
            step_count = 0
            step_rewards = [] 
            
            for agent in env.agent_iter():
                obs, reward, termination, truncation, info = env.last()
                
                step_reward = sum(env.rewards.values()) / n_agents
                step_rewards.append(step_reward)
            
                for a in env.agents:
                    current_rewards[a] += env.rewards[a]
                
                step_count += 1
                
                if termination or truncation:
                    break
                else:
                    act = model.predict(obs, deterministic=deterministic)[0]
                    env.step(act)

            per_step_rewards.append(step_rewards)

            
            # After each episode
            avg_episode_reward = sum(current_rewards.values()) / n_agents
            episode_rewards.append(avg_episode_reward)
            episode_lengths.append(step_count)
            
            for agent, reward in current_rewards.items():
                all_agent_rewards[agent].append(reward)
            
            print(f"Episode {i + 1} - Average Reward: {avg_episode_reward:.2f}, Length: {step_count}")
            for agent, reward in current_rewards.items():
                print(f"  {agent} Reward: {reward:.2f}")
            
            # Write to CSV
            csv_row = [i+1, avg_episode_reward, step_count] + [current_rewards[agent] for agent in env.possible_agents]
            csvwriter.writerow(csv_row)

    env.close()

    avg_reward = np.mean(episode_rewards)
    avg_length = np.mean(episode_lengths)
    
    print("\nEvaluation Results:")
    print(f"Average Reward over {num_games} episodes: {avg_reward:.2f}")
    print(f"Average Episode Length: {avg_length:.2f}")
    print(f"Reward Standard Deviation: {np.std(episode_rewards):.2f}")
    print(f"Results saved to {csv_path}")

    return episode_rewards, all_agent_rewards, per_step_rewards, csv_path

# Usage

    

##callback add - can see at which step, all plots?, eval and monitor wrapper:
# Evaluate the model periodically
# and auto-save the best model and evaluations
# Use a monitor wrapper to properly report episode stats
#eval_env = Monitor(gym.make("LunarLander-v2"))
# Use deterministic actions for evaluation
#eval_callback = EvalCallback(eval_env, best_model_save_path="./logs/",
#                             log_path="./logs/", eval_freq=2000,
#                             deterministic=True, render=False)
#model.learn(total_timesteps=20000, callback=[checkpoint_callback, eval_callback])
#https://araffin.github.io/post/sb3/

def call_eval(n_evaluations = 5):
        
    # Run multiple evaluations to get a distribution
    all_rewards = []
    all_agent_rewards = {agent: [] for agent in env_fn.env().possible_agents}
    all_per_step_rewards = []

    for _ in range(n_evaluations):
        print(str(n_evaluations) + " eval start" ) 
        rewards, agent_rewards, per_step_rewards, csv_path = eval_new(_, env_fn, num_games=10, render_mode=None, save_path="./eval_results", **env_kwargs)
        all_rewards.append(rewards)
        for agent, rewards in agent_rewards.items():
            all_agent_rewards[agent].append(rewards)
        all_per_step_rewards.extend(per_step_rewards)
    # Plotting
    fig, axes = plt.subplots(1, 5, figsize=(20, 6))

    #plot_with_confidence_interval(axes[0], all_rewards, "Average Reward", "blue")
    axes[0].set_title("Average Episode Rewards")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Reward")

    # Agent-wise Rewards
    colors = plt.cm.rainbow(np.linspace(0, 1, len(all_agent_rewards)))
    for (agent, rewards), color in zip(all_agent_rewards.items(), colors):
        plot_with_confidence_interval(axes[1], rewards, agent, color)
        axes[1].set_title("Agent-wise Rewards")
        axes[1].set_xlabel("Episode")
        axes[1].legend()
        axes[1].set_ylabel("Reward")

    # Plot rewards over time for two random episodes
    random_episodes = np.random.choice(len(all_per_step_rewards), 3, replace=False)
    for i, episode_num in enumerate(random_episodes):
        plot_episode_rewards(axes[i+2], all_per_step_rewards[episode_num], episode_num + 1)

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(csv_path), "eval_results_plot_with_episodes.png"))
    plt.show()


if __name__ == "__main__":
    env_fn = waterworld_model1
    env_kwargs = {}

    # Train a model (takes ~3 minutes on GPU)
    #train_butterfly_supersuit(env_fn, steps=196_608, seed=0, **env_kwargs)

    # Evaluate 10 games (average reward should be positive but can vary significantly)
    log_dir = train_butterfly_supersuit(env_fn, steps=196_608, seed=0, **env_kwargs)
    
    # Plot the results
    from stable_baselines3.common import results_plotter
    results_plotter.plot_results(
        [log_dir], 196_608, results_plotter.X_TIMESTEPS, "PPO Waterworld"
    )

    # Watch 2 games    #eval(env_fn, num_games=10, render_mode=None, **env_kwargs)
    #call_eval()
    #plot_results_custom(log_dir)
    
    


    