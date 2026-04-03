import os
import matplotlib.pyplot as plt
from multi_arm_bandit import bandit_algorithm
import numpy as np

np.random.seed(0)
steps = 10000
runs = 2000
arms = 10
epsilon = 0.1
reward_std = 0.01
ucb_constant = 2
true_values = np.random.randn(arms)
print("True values:", true_values)

def main():
    # Create the 'figures' directory if7 it doesn't exist for figures storage
    os.makedirs('multi_armed_bandits/figures', exist_ok=True)

    # fig 1.1: epsilon-greedy action selection (stationary bandit)
    # runs with sample-average step-size
    stationary_rewards_SAM, stationary_optimal_SAM, stationary_label_SAM = bandit_algorithm(steps=steps, runs=runs, arms=arms, epsilon=epsilon, true_values=true_values,step_size=None, reward_std=reward_std)
    # runs with constant step-size
    stationary_rewards_CSS, stationary_optimal_CSS, stationary_label_CSS = bandit_algorithm(steps=steps, runs=runs, arms=arms, epsilon=epsilon, true_values=true_values, step_size=0.1, reward_std=reward_std)
    plt.figure()
    # plot average rewards
    plt.subplot(2,1,1)
    plt.plot(np.mean(stationary_rewards_SAM,axis = 1), label = stationary_label_SAM)
    plt.plot(np.mean(stationary_rewards_CSS,axis = 1), label = stationary_label_CSS)
    plt.legend()
    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    # plot % optimal actions
    plt.subplot(2,1,2)
    plt.plot(np.mean(stationary_optimal_SAM,axis = 1), label = stationary_label_SAM)
    plt.plot(np.mean(stationary_optimal_CSS,axis = 1), label = stationary_label_CSS)
    plt.ylabel("% Optimal Action")
    plt.xlabel("Steps")
    plt.legend()
    plt.suptitle(f"Figure 1.1: Epsilon-Greedy Action Selection for Stationary Bandits({runs} runs),\n With Epsilon={epsilon} & Reward~N(true value, {reward_std**2:.2f})",
                 fontsize=10)
    plt.savefig(os.path.join('multi_armed_bandits/figures','exercise2-5-fig1-1.png'))

    # fig 1.2: UCB action selection (stationary values)
    # runs with sample-averaging step-size
    stationary_rewards_SAM_UCB, stationary_optimal_SAM_UCB, stationary_label_SAM_UCB = bandit_algorithm(steps=steps, runs=runs, arms=arms, ucb_constant=ucb_constant, true_values=true_values, step_size=None, reward_std=reward_std)
    # runs with constant step-size
    stationary_rewards_CSS_UCB, stationary_optimal_CSS_UCB, stationary_label_CSS_UCB = bandit_algorithm(steps=steps, runs=runs, arms=arms, ucb_constant=ucb_constant, true_values=true_values, step_size=0.1, reward_std=reward_std)
    plt.figure()
    # plot average rewards
    plt.subplot(2,1,1)
    plt.plot(np.mean(stationary_rewards_SAM_UCB,axis = 1), label = stationary_label_SAM_UCB)
    plt.plot(np.mean(stationary_rewards_CSS_UCB,axis = 1), label = stationary_label_CSS_UCB)
    plt.legend()
    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    # plot % optimal actions
    plt.subplot(2,1,2)
    plt.plot(np.mean(stationary_optimal_SAM_UCB,axis = 1), label = stationary_label_SAM_UCB)
    plt.plot(np.mean(stationary_optimal_CSS_UCB,axis = 1), label = stationary_label_CSS_UCB)
    plt.ylabel("% Optimal Action")
    plt.xlabel("Steps")
    plt.legend()
    plt.suptitle(f"Figure 1.2: UCB Action Selection for Stationary Rewards ({runs} runs),\n With UCB Constant={ucb_constant} & Reward~N(true value, {reward_std**2:.2f})",
                 fontsize=10)
    plt.savefig(os.path.join('multi_armed_bandits/figures','exercise2-5-fig1-2.png'))

    # fig 1.3: Gradient bandit algorithm (stationary values)
    # runs with step-size=0.1
    stationary_rewards_gradient1, stationary_optimal_gradient1, stationary_label_gradient1 = bandit_algorithm(steps=steps, runs=runs, arms=arms, gradient_bandit=True, true_values=true_values, step_size=0.1, reward_std=reward_std)
    # runs with step-size=0.4
    stationary_rewards_gradient2, stationary_optimal_gradient2, stationary_label_gradient2 = bandit_algorithm(steps=steps, runs=runs, arms=arms, gradient_bandit=True, true_values=true_values, step_size=0.4, reward_std=reward_std)
    plt.figure()
    # plot average rewards
    plt.subplot(2,1,1)
    plt.plot(np.mean(stationary_rewards_gradient1,axis = 1), label = stationary_label_gradient1)
    plt.plot(np.mean(stationary_rewards_gradient2,axis = 1), label = stationary_label_gradient2)
    plt.legend()
    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    # plot % optimal actions
    plt.subplot(2,1,2)
    plt.plot(np.mean(stationary_optimal_gradient1,axis = 1), label = stationary_label_gradient1)
    plt.plot(np.mean(stationary_optimal_gradient2,axis = 1), label = stationary_label_gradient2)
    plt.ylabel("% Optimal Action")
    plt.xlabel("Steps")
    plt.legend()
    plt.suptitle(f"Figure 1.3: Gradient Bandit Algorithm for Stationary Rewards ({runs} runs),\n Reward~N(true value, {reward_std**2:.2f})",
                 fontsize=10)
    plt.savefig(os.path.join('multi_armed_bandits/figures','exercise2-5-fig1-3.png'))

    # fig 1.4: Algorithms Overview for Stationary Rewards
    plt.figure()
    # plot average rewards
    plt.subplot(2,1,1)
    # 1.1
    plt.plot(np.mean(stationary_rewards_SAM,axis = 1), label = stationary_label_SAM)
    plt.plot(np.mean(stationary_rewards_CSS,axis = 1), label = stationary_label_CSS)
    # 1.3
    plt.plot(np.mean(stationary_rewards_gradient1,axis = 1), label = stationary_label_gradient1)
    plt.plot(np.mean(stationary_rewards_gradient2,axis = 1), label = stationary_label_gradient2)
    plt.legend()
    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    # plot % optimal actions
    plt.subplot(2,1,2)
    # 1.1
    plt.plot(np.mean(stationary_optimal_SAM,axis = 1), label = stationary_label_SAM)
    plt.plot(np.mean(stationary_optimal_CSS,axis = 1), label = stationary_label_CSS)
    # 1.3
    plt.plot(np.mean(stationary_optimal_gradient1,axis = 1), label = stationary_label_gradient1)
    plt.plot(np.mean(stationary_optimal_gradient2,axis = 1), label = stationary_label_gradient2)
    plt.ylabel("% Optimal Action")
    plt.xlabel("Steps")
    plt.legend()
    plt.suptitle(f"Figure 1.4: Algorithms Overview for Stationary Rewards ({runs} runs),\n Reward~N(true value, {reward_std**2:.2f})",
                 fontsize=10)
    plt.savefig(os.path.join('multi_armed_bandits/figures','exercise2-5-fig1-4.png'))


    # fig 2.1: epsilon-greedy action selection (nonstationary values)
    # runs with sample-average step-size
    nonstationary_rewards_SAM, nonstationary_optimal_SAM, nonstationary_label_SAM = bandit_algorithm(steps=steps, runs=runs, arms=arms, epsilon=epsilon, true_values=true_values, stationary_values=False, step_size=None, reward_std=reward_std)
    # runs with constant step-size
    nonstationary_rewards_CSS, nonstationary_optimal_CSS, nonstationary_label_CSS = bandit_algorithm(steps=steps, runs=runs, arms=arms, epsilon=epsilon, true_values=true_values, stationary_values=False, step_size=0.1, reward_std=reward_std)
    plt.figure()
    # plot average rewards
    plt.subplot(2,1,1)
    plt.plot(np.mean(nonstationary_rewards_SAM,axis = 1), label = nonstationary_label_SAM)
    plt.plot(np.mean(nonstationary_rewards_CSS,axis = 1), label = nonstationary_label_CSS)
    plt.legend()
    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    # plot % optimal actions
    plt.subplot(2,1,2)
    plt.plot(np.mean(nonstationary_optimal_SAM,axis = 1), label = nonstationary_label_SAM)
    plt.plot(np.mean(nonstationary_optimal_CSS,axis = 1), label = nonstationary_label_CSS)
    plt.ylabel("% Optimal Action")
    plt.xlabel("Steps")
    plt.legend()
    plt.suptitle(f"Figure 2.1: Epsilon-Greedy Action Selection for Nonstationary Rewards ({runs} runs),\n With Epsilon={epsilon} & Reward~N(true value, {reward_std**2:.2f})",
                 fontsize=10)
    plt.savefig(os.path.join('multi_armed_bandits/figures','exercise2-5-fig2-1.png'))

    # fig 2.2: UCB action selection (nonstationary values)
    # runs with sample-averaging step-size
    nonstationary_rewards_SAM_UCB, nonstationary_optimal_SAM_UCB, nonstationary_label_SAM_UCB = bandit_algorithm(steps=steps, runs=runs, arms=arms, ucb_constant=ucb_constant, true_values=true_values, stationary_values=False, step_size=None, reward_std=reward_std)
    # runs with constant step-size
    nonstationary_rewards_CSS_UCB, nonstationary_optimal_CSS_UCB, nonstationary_label_CSS_UCB = bandit_algorithm(steps=steps, runs=runs, arms=arms, ucb_constant=ucb_constant, true_values=true_values, stationary_values=False, step_size=0.1, reward_std=reward_std)
    plt.figure()
    # plot average rewards
    plt.subplot(2,1,1)
    plt.plot(np.mean(nonstationary_rewards_SAM_UCB,axis = 1), label = nonstationary_label_SAM_UCB)
    plt.plot(np.mean(nonstationary_rewards_CSS_UCB,axis = 1), label = nonstationary_label_CSS_UCB)
    plt.legend()
    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    # plot % optimal actions
    plt.subplot(2,1,2)
    plt.plot(np.mean(nonstationary_optimal_SAM_UCB,axis = 1), label = nonstationary_label_SAM_UCB)
    plt.plot(np.mean(nonstationary_optimal_CSS_UCB,axis = 1), label = nonstationary_label_CSS_UCB)
    plt.ylabel("% Optimal Action")
    plt.xlabel("Steps")
    plt.legend()
    plt.suptitle(f"Figure 2.2: UCB Action Selection for Nonstationary Rewards ({runs} runs),\n With UCB Constant={ucb_constant} & Reward~N(true value, {reward_std**2:.2f})",
                 fontsize=10)
    plt.savefig(os.path.join('multi_armed_bandits/figures','exercise2-5-fig2-2.png'))

    # fig 2.3: Gradient Bandit Algorithm (nonstationary values)
    # runs with step-size=0.1
    nonstationary_rewards_gradient1, nonstationary_optimal_gradient1, nonstationary_label_gradient1 = bandit_algorithm(steps=steps, runs=runs, arms=arms, gradient_bandit=True, true_values=true_values, stationary_values=False, step_size=0.1, reward_std=reward_std)
    # runs with step-size=0.4
    nonstationary_rewards_gradient2, nonstationary_optimal_gradient2, nonstationary_label_gradient2 = bandit_algorithm(steps=steps, runs=runs, arms=arms, gradient_bandit=True, true_values=true_values, stationary_values=False, step_size=0.4, reward_std=reward_std)
    plt.figure()
    # plot average rewards
    plt.subplot(2,1,1)
    plt.plot(np.mean(nonstationary_rewards_gradient1,axis = 1), label = nonstationary_label_gradient1)
    plt.plot(np.mean(nonstationary_rewards_gradient2,axis = 1), label = nonstationary_label_gradient2)
    plt.legend()
    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    # plot % optimal actions
    plt.subplot(2,1,2)
    plt.plot(np.mean(nonstationary_optimal_gradient1,axis = 1), label = nonstationary_label_gradient1)
    plt.plot(np.mean(nonstationary_optimal_gradient2,axis = 1), label = nonstationary_label_gradient2)
    plt.ylabel("% Optimal Action")
    plt.xlabel("Steps")
    plt.legend()
    plt.suptitle(f"Figure 2.3: Gradient Bandit Algorithm for Nonstationary Rewards ({runs} runs),\n Reward~N(true value, {reward_std**2:.2f})",
                 fontsize=10)
    plt.savefig(os.path.join('multi_armed_bandits/figures','exercise2-5-fig2-3.png'))

    # fig 2.4: Algorithms Overview (nonstationary values)
    plt.figure()
    # plot average rewards
    plt.subplot(2,1,1)
    # 2.1
    plt.plot(np.mean(nonstationary_rewards_SAM,axis = 1), label = nonstationary_label_SAM)
    plt.plot(np.mean(nonstationary_rewards_CSS,axis = 1), label = nonstationary_label_CSS)
    # 2.3
    plt.plot(np.mean(nonstationary_rewards_gradient1,axis = 1), label = nonstationary_label_gradient1)
    plt.plot(np.mean(nonstationary_rewards_gradient2,axis = 1), label = nonstationary_label_gradient2)
    plt.legend()
    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    # plot % optimal actions
    plt.subplot(2,1,2)
    # 2.1
    plt.plot(np.mean(nonstationary_optimal_SAM,axis = 1), label = nonstationary_label_SAM)
    plt.plot(np.mean(nonstationary_optimal_CSS,axis = 1), label = nonstationary_label_CSS)
    # 2.3
    plt.plot(np.mean(nonstationary_optimal_gradient1,axis = 1), label = nonstationary_label_gradient1)
    plt.plot(np.mean(nonstationary_optimal_gradient2,axis = 1), label = nonstationary_label_gradient2)
    plt.ylabel("% Optimal Action")
    plt.xlabel("Steps")
    plt.legend()
    plt.suptitle(f"Figure 2.4: Algorithms Overview for Nonstationary Rewards ({runs} runs),\n Reward~N(true value, {reward_std**2:.2f})",
                 fontsize=10)
    plt.savefig(os.path.join('multi_armed_bandits/figures','exercise2-5-fig2-4.png'))
if __name__ == "__main__":
    main()
