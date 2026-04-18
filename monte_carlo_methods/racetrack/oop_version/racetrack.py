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

def off_policy_control(Racetrack, gamma=0.9, epsilon=0.1, max_episode_count=1000, max_steps=100000):
    # cumulative sum sampling ratio: state = 4d, action = 3d; total 7d
    cum_IS = np.zeros((len(Racetrack.racetrack),len(Racetrack.racetrack[0]),5,5,3,3),int)
    episode_count = 0    
    behaviour_policy = utils.get_policy(Racetrack, epsilon=epsilon)
    while True:        
        episode = utils.Episode(Racetrack, behaviour_policy)
        if episode.generate(Racetrack, max_steps=100000):
            episode_count += 1
            incremental_prediction(episode, cum_IS, gamma, epsilon)
            if episode_count > max_episode_count:
                behaviour_policy = utils.get_policy(Racetrack, epsilon=epsilon)
                break
            print(f"Episode {episode_count} complete")
        else:
            print(f'Episode generation reached maximum steps ({max_steps}), retrying...')
            break


def main():
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
                   max_steps=100000)
    

main()