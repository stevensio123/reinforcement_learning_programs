import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from multi_arm_bandit import bandit_algorithm


epsilon_parameters = np.logspace(-3, 0, 10) # from 1e-3 to 1.0 in logarithmic scale
print(f"Epsilon Parameters: {epsilon_parameters}")

step_size_parameters = np.logspace(-3, 1, 10) # from 1e-3 to 10 in logarithmic scale
print(f"Step-size Parameters: {step_size_parameters}")

np.random.seed(0)
steps = 10000
last_steps = 5000
runs = 200
arms = 10
reward_std = 0.1
random_walk_variance = 0.01
stationary_bool = True
true_values = np.random.randn(arms)
print("True values:", true_values)

# learning curve 1: constant-step-size epsilon-greedy with different epsilon values and alpha=0.1
epsilon_avg_performance = []
for parameter in epsilon_parameters:    
    epsilon_rewards, _, _ = bandit_algorithm(steps=steps,
                                             runs=runs,
                                             arms=arms,
                                             epsilon=parameter, 
                                             true_values=true_values, 
                                             stationary_values=stationary_bool,
                                             random_walk_variance=random_walk_variance, 
                                             step_size=0.1, 
                                             reward_std=reward_std)
    # get per run average reward for the last n steps
    per_run_mean = np.mean(epsilon_rewards[-last_steps:, :], axis=0) # shape (runs,)
    # get mean over all runs
    epsilon_avg_performance.append(per_run_mean.mean())
    # to get grand average reward directly:
    # np.mean(rewards[-last_steps:, :]) 
    
parameter_study_df = pd.DataFrame({'parameter_name':'Epsilon',
                                   'parameter':epsilon_parameters,
                                   'avg_reward':epsilon_avg_performance})


# learning curve 2: constant-step-size UCB algorithm with different confidence levels and alpha=0.1
ucb_avg_performance = []
for parameter in step_size_parameters:    
    ucb_rewards, _, _ = bandit_algorithm(steps=steps,
                                         runs=runs,
                                         arms=arms,
                                         epsilon=None,
                                         ucb_constant=parameter, 
                                         true_values=true_values, 
                                         stationary_values=stationary_bool,
                                         random_walk_variance=random_walk_variance, 
                                         step_size=0.1, 
                                         reward_std=reward_std)
    # get per run average reward for the last n steps
    per_run_mean = np.mean(ucb_rewards[-last_steps:, :], axis=0) # shape (runs,)
    # get mean over all runs
    ucb_avg_performance.append(per_run_mean.mean())

ucb_df = pd.DataFrame({'parameter_name':'UCB-confidence level',
                       'parameter':step_size_parameters,
                       'avg_reward':ucb_avg_performance})
parameter_study_df = pd.concat([parameter_study_df, ucb_df], ignore_index=True)

# learning curve 3: gradient-bandit algorithm with different step-sizes
gradient_avg_performance = []
for parameter in step_size_parameters:    
    gradient_rewards, _, _ = bandit_algorithm(steps=steps,
                                              runs=runs,
                                              arms=arms,
                                              epsilon=None,
                                              ucb_constant=None,
                                              gradient_bandit=True, 
                                              true_values=true_values, 
                                              stationary_values=stationary_bool,
                                              random_walk_variance=random_walk_variance, 
                                              step_size=parameter, 
                                              reward_std=reward_std)
    # get per run average reward for the last n steps
    per_run_mean = np.mean(gradient_rewards[-last_steps:, :], axis=0) # shape (runs,)
    # get mean over all runs
    gradient_avg_performance.append(per_run_mean.mean())

gradient_df = pd.DataFrame({'parameter_name':'Gradient-bandit step-size',
                       'parameter':step_size_parameters,
                       'avg_reward':gradient_avg_performance})
parameter_study_df = pd.concat([parameter_study_df, gradient_df], ignore_index=True)

# plot parameter study results
sns.lineplot(data=parameter_study_df, x='parameter', y='avg_reward', hue='parameter_name', marker='o')
plt.xscale('log')
plt.xlabel('Parameter Value (log scale)')
plt.ylabel(f'Average Reward over last {last_steps} steps of {steps} total steps')
plt.title(f'Average Reward vs Parameter Value for Different Bandit Algorithms\nwith stationary true values (averaged over {runs} runs).')
plt.legend(title='Parameter Name')
plt.grid(True)
plt.tight_layout()
plt.savefig('multi_armed_bandits/figures/exercise2_11_parameter_study_v1.png')
print('Done')
