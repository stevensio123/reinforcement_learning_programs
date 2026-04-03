
# import math
import matplotlib.pyplot as plt
# import matplotlib.axes as axes
import numpy as np
import pandas as pd
# import seaborn as sns
from scipy.stats import poisson

# poisson parameters
mean_return_a = 4
mean_rental_a = 3
mean_return_b = 2
mean_rental_b = 4

# rental and return upper limits
upper_return_a = 9
upper_rental_a = 9
upper_return_b = 9
upper_rental_b = 9

# used to define reward space later
upper_reward = (upper_rental_a + upper_rental_b) * 10

# discount parameter
gamma = 0.9

# random seed
np.random.seed(42)

# small positive number determining accuracy of policy estimation
theta = 0.1

# action and reward space
action_space = np.arange(-5,6,1)


# Setting reward space to upper limit
reward_space = np.arange(0, upper_reward + 1, 10)

# state value dict
state_space = []
for carsA in range(0,21):
    for carsB in range(0,21):
      state_space.append((carsA,carsB))

# next state value dict
next_state_space = []
for nextcarsA in range(0,30):
    for nextcarsB in range(0,30):
      next_state_space.append((nextcarsA,nextcarsB))

# initialize with random values
# e.g. policy_dict[(0,0)] = -5  <- this should not be possible in the first place
state_values = np.random.normal(0,1,size=len(state_space))
state_values_dict = dict(zip(state_space, state_values))
policy_dict = dict(zip(state_space, np.random.randint(-5,6,size=len(state_space))))


'''
# sanity checking
print(f'total states: {len(state_space_index)}')
print(f'state index: {state_space_index[0:5]}')
print(f'states: {state_space[0:5]}')
print(f'state values: {state_values[:5]}')
print(f'state value 1 from dict: {state_values_dict[(0,1)]}')
print(f'action space: {action_space[:]}')
print(f'reward space: {reward_space[:5]}')
print(f'policy for state (0,1): {policy_dict[10,20]}')
'''
def interm_state(s,action):
  # function (s,a) --> intermstate
  a = s[0] - action
  b = s[1] + action

  return (a, b)

def q(s,a):

  q = 0
  # Generates new POSSIBLE rewards space based off action taken
  next_reward_space = []

  # check the interim state after action
  interm = interm_state(s,a)

  # adjust new possible reward space
  for r in reward_space:

    # For exercise 4.7 part 1
    if a > 0:
      r += (a * -2) + 2
    else:
      r += (abs(a) * -2)

    #For exercise 4.7 part 2
    if interm[0] > 10:
      r -= 4
    if interm[1] > 10:
      r -= 4

    next_reward_space.append(int(r))

  # loop 19 times
  for r_index, r in enumerate(next_reward_space):

    # loop 441 times
    for next in next_state_space:
      # state_value_dict[next] gives us the quick lookup of the next state's state value
      probability = p(next,r_index,interm)
      if next[0] > 20:
        next = (20, next[1])
      if next[1] > 20:
        next = (next[0], 20)
      q += probability * (r + (gamma * state_values_dict[next]))

  return q

def p(next,r_index,interm):

  # initiate required probability variable
  final_probability = 0

  for a_rental in range(0,int(r_index + 1)):
    if a_rental > min(upper_rental_a, interm[0]):
      continue

    b_rental = r_index - a_rental

    if b_rental > min(upper_rental_b, interm[1]):
      continue

    a_return = (next[0] - (interm[0] - a_rental))
    if a_return > upper_return_a:
      continue

    b_return = (next[1] - (interm[1] - b_rental))
    if b_return > upper_return_b:
      continue

    if a_return < 0 or b_return < 0:
      continue

    # use array indexing to find probability
    final_probability += P[int(a_rental),int(a_return),int(b_rental),int(b_return)]
  return final_probability

def plot_policy_values(policy_name):
  s=pd.Series(state_values_dict)
  df = s.unstack()

  # enforce exact order 0..20 on both axes (optional)
  order = list(range(21))
  df = df.reindex(index=order, columns=order)

  arr = df.to_numpy()

  fig, ax = plt.subplots()

  im = ax.imshow(arr, origin='lower')  # bottom-left is (0,0)

  ticks = np.arange(0, 21, 1) # tick labels with interval of 1
  ax.set_xticks(ticks)
  ax.set_yticks(ticks)
  ax.set_xticklabels(ticks)
  ax.set_yticklabels(ticks)
  ax.set_xlabel("location_b")
  ax.set_ylabel("location_a")

  # colorbar = legend for heatmap values
  cbar = fig.colorbar(im, ax=ax)
  cbar.set_label("value")

  plt.title(policy_name)
  # SAVE TO PNG
  plt.savefig(f"{policy_name}.png", dpi=300, bbox_inches='tight')
  #plt.show()
  plt.close()
  plt.show()

def policy_evaluation():

  eval_counter = 0
  while True:
    delta = 0
    for s in state_space:
      old_v = state_values_dict[s]
      a = policy_dict[s]
      new_v = q(s, a)
      state_values_dict[s] = new_v
      delta = max(delta, np.abs(old_v - new_v))

    if delta < theta:
      break
    print(f"evaluated round: {eval_counter},\n"
          f"last delta = {delta} still greater than theta ({theta}), continuing")
    eval_counter += 1
  print(f"\npolicy evaluation complete, last delta = {delta}\n")

def plot_policy(policy_name):
  x, y = zip(*policy_dict.keys())
  df = pd.DataFrame({'location_1': list(x), 'location_2': list(y), 'action': list(policy_dict.values())})

  df_pivot = df.pivot(index='location_1', columns='location_2', values='action')
  levels = [-5,-4,-3,-2,-1,0,1,2,3,4,4.9]

  cs = plt.contour(
      df_pivot.columns,      # X axis values (location_2)
      df_pivot.index,        # Y axis values (location_1)
      df_pivot.values,       # Z values (action)
      linewidths=0.5,
      levels=levels,
      norm=plt.Normalize(vmin=-5, vmax=5))

  fmt = {}
  strs = [-5,-4,-3,-2,-1,0,1,2,3,4,5]
  for l,s in zip( cs.levels, strs ):
      fmt[l] = s
  plt.clabel(cs, inline=True, fmt=fmt, fontsize=8)

  plt.xlabel('location_b')
  plt.ylabel('location_a')

  plt.xticks(range(int(df_pivot.columns.min()), int(df_pivot.columns.max())+1, 1))
  plt.yticks(range(int(df_pivot.index.min()), int(df_pivot.index.max())+1, 1))
  plt.title(policy_name)
  # SAVE TO PNG
  plt.savefig(f"{policy_name}.png", dpi=300, bbox_inches='tight')
  plt.close()

def policy_improvement():
  policy_stable = True
  for s in state_space:
    old_action = policy_dict[s]
    # update state_values_dict
    q_list=[]
    for a in action_space:
      if a > s[0] or -a > s[1]:
        q_list.append(0)
      else:
        q_list.append(q(s,a))

    policy_dict[s] = action_space[np.argmax(q_list)]
    if old_action != action_space[np.argmax(q_list)]:
      policy_stable = False
  return policy_stable

def policy_iteration():
  policy_counter = 0

  # initial policy plots
  plot_policy_values(f"policy-{policy_counter}-state-values")
  plot_policy(f"policy-{policy_counter}")
  policy_stable = False

  while True:
    policy_counter += 1
    print("policy not stable\n")
    print(f"beginning policy iteration round {policy_counter}\n")
    print("running policy evaluation...")
    policy_evaluation()
    plot_policy_values(f"policy-{policy_counter}-state-values")
    print("running policy improvement...")
    policy_stable = policy_improvement()
    print("policy improved...\n")
    # plot policy after improvement here, figures saved to home directory
    plot_policy(f"policy-{policy_counter}")

    # break the loop if true
    if policy_stable == True:
      break

  print("policy stable")

def main():
  policy_iteration()

main()
