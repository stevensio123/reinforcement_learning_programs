import numpy as np
import racetrack_utils as utils

def incremental_prediction(Racetrack, episode, cum_IS, gamma=0.9, epsilon=0.1):
    
    episode.reverse()
    G = 0
    W = 1
    for step in range(len(episode)):
        G = (gamma*G) - 1
        x,y,v1,v2 = episode[step][0] # state
        a1,a2 = episode[step][1] # action
        cum_IS[x,y,v1,v2,a1,a2] += (cum_IS[x,y,v1,v2,a1,a2] + W)
        next_state = utils.get_next_state(Racetrack.racetrack, (x,y,v1,v2), (a1,a2))
        next_state_value = Racetrack.get_state_value(next_state)
        Racetrack.state_values[x,y,v1,v2] = next_state_value + (W/cum_IS[x,y,v1,v2,a1,a2]) * (G - next_state_value)
        action_space_ls = utils.get_action_space((x,y,v1,v2))
        optimal_action_idx = utils.get_optimal_action((x,y,v1,v2), Racetrack, action_space_ls)
        optimal_action = action_space_ls[optimal_action_idx]
        if (a1,a2) != optimal_action:
            break
        W = W / ((1- epsilon) + (epsilon / len(action_space_ls)))

def off_policy_control(Racetrack, gamma=0.9, epsilon=0.1, max_episode_count=1000, max_steps=100000, max_episode_generation_attempts=100):
    # cumulative sum sampling ratio: state = 4d, action = 3d; total 7d
    cum_IS = np.zeros((len(Racetrack.racetrack),len(Racetrack.racetrack[0]),5,5,3,3),int)
    episode_count = 0    
    behaviour_policy = utils.get_policy(Racetrack, epsilon=epsilon)
    ep_generation_attempts = 0
    while True:        
        episode = utils.Episode(Racetrack, behaviour_policy)
        if episode.generate(Racetrack, max_steps=max_steps):
            episode_count += 1
            incremental_prediction(episode, cum_IS, gamma, epsilon)
            if episode_count > max_episode_count:
                print(f"Off-policy control complete after {episode_count} episodes")
                break
            print(f"Episode {episode_count} complete")
        else:
            ep_generation_attempts += 1
            print(f'Episode generation reached maximum steps ({max_steps}), retrying...')
            print(f"Episode generation attempts: {ep_generation_attempts}")
            if ep_generation_attempts > max_episode_generation_attempts:
                if epsilon < 1: # or <= 0.99
                    epsilon += 0.01
                    print(f"Episode generation failed after {ep_generation_attempts} attempts, increased epsilon by 0.01./n New epsilon: {epsilon:.2f}")
                else:
                    print(f"Episode generation failed after {ep_generation_attempts} attempts, epsilon is already at maximum value of 1. Stopping off-policy control.")
                    break
                ep_generation_attempts = 0
            behaviour_policy = utils.get_policy(Racetrack, epsilon=epsilon)
            pass

def main():
    np.random.seed(42)
    race_track = ["####EEEE",
                "#NNNNNNE",
                "#NNNNNNE",
                "#NNNNNNE",
                "#NNNNN##",
                "#SSSSS##"]
    '''
    race_track = ["########",
                "#NNNNNNE",
                "#NN#####",
                "#NN#####",
                "#SS#####"]
    '''
    gamma = 0.9
    max_episode_count = 1000
    race_track_obj = utils.Racetrack(race_track)

    off_policy_control(race_track_obj,
                   gamma=gamma,
                   epsilon=0.1, 
                   max_episode_count=max_episode_count,
                   max_steps=1000000,
                   max_episode_generation_attempts=10)
    

main()