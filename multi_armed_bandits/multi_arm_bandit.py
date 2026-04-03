import numpy as np

def bandit_algorithm(steps, runs, arms, true_values, stationary_values=True, gradient_bandit=False, epsilon=None, ucb_constant=None, step_size=None, initial_estimate=0.0, reward_std=0.01, random_walk_variance=0.01):
    # check some parameters requirements
    if epsilon is None and ucb_constant is None and gradient_bandit==False:
        raise ValueError("If gradient_bandit is False, must specify either epsilon or UCB constant parameter")
    if (epsilon is not None) and (ucb_constant is not None):
        raise ValueError("Only one of the parameters should be specified: UCB constant or epsilon")
    if gradient_bandit and ((epsilon is not None) or (ucb_constant is not None)):
        raise ValueError("If gradient_bandit is True, cannot use UCB or epsilon-greedy algorithm.")
    if (epsilon is not None and epsilon < 0) or (ucb_constant is not None and ucb_constant < 0):
        raise ValueError("Specified epsilon or UCB constant must be equal or above zero")
    # empty 2D arrays to store each step's result
    step_reward = np.empty(shape=[steps,runs])
    step_optimal = np.empty(shape=[steps,runs])
    for run in range(runs):
        # reset values for each run
        # reward = []
        # reward.append(0)
        true_values_run = np.copy(true_values)
        estimated_values = np.ones(arms) * initial_estimate
        preference = np.zeros(arms)
        baseline = 0.0
        n = np.zeros(arms).astype(int)

        for step in range(steps):
            # ACTION SELECTION METHODS
            # use epsilon-greedy action selection
            if (epsilon is not None): 
                if np.random.rand() <= epsilon:
                    action = np.random.randint(arms)
                elif np.max(estimated_values) == 0.0:
                    action = np.random.randint(arms)
                else:
                    action = np.argmax(estimated_values)
            # use gradient bandit action selection
            elif gradient_bandit: 
                exp_preferences = np.exp(preference - np.max(preference))  # sift downward for numerical stability
                action_probabilities = exp_preferences / np.sum(exp_preferences)
                # action_probabilities = np.exp(preference) / np.sum(np.exp(preference)) # original formula
                action = np.random.choice(np.arange(arms), p=action_probabilities)
            # use UCB action selection
            elif (ucb_constant is not None):
                # start with round-robin (try every arm once)
                if step < arms:
                    action = step # round-robin 
                else:
                    ucb_values = estimated_values + ucb_constant * np.sqrt(np.log(step + 1) / (n + 1e-5))
                    action = np.argmax(ucb_values)
            else:
                raise ValueError("No action selection method detected")        
            
            # REWARD FUNCTIONS
            if stationary_values: # select reward based on normal probabilities of the constant true values
                new_reward = true_values_run[action] + np.random.normal(0, reward_std)
                # reward.append(new_reward)
            else: # update true values with random walk, then select reward based on normal probabilities.
                true_values_run += np.random.normal(0,random_walk_variance, size=arms)
                new_reward = true_values_run[action] + np.random.normal(0, reward_std)
                # reward.append(new_reward)                
            step_reward[step,run] = new_reward

            # FILL UP RESULTS ARRAYS
            if action == np.argmax(true_values_run):
                step_optimal[step,run] = 1
            else:
                step_optimal[step,run] = 0

            # UPDATE ESTIMATED VALUES / PREFERENCES
            n[action] += 1
            if step_size is None:
                # use sample-averages of observed rewards
                estimated_values[action] += (new_reward - estimated_values[action]) / n[action]
            elif gradient_bandit:
                # use gradient bandit update rule
                indices = np.arange(len(preference))
                mask = (indices == action) # boolean mask: True only at index == action, False otherwise
                # update preference for selected action
                preference[mask] += step_size * (new_reward - baseline) * (1 - action_probabilities[action])
                preference[~mask] -= step_size * (new_reward - baseline) * action_probabilities[~mask]
                # using incremental formula to update baseline (same as reward average)
                baseline += (new_reward - baseline) / (step + 1)
            else:
                # use constant step-size parameter
                estimated_values[action] += (new_reward - estimated_values[action]) * step_size
                
    if step_size is None:
        label = "Sample Averaging (step size=1/n) "
    elif gradient_bandit:
        label = f"Gradient Bandit, step-size = {step_size}"
    else:
        label = f"Constant-step-size = {step_size}"

    return step_reward, step_optimal, label
