import numpy as np

OBS_SCALE = 1
REWARD_WEIGHTS = [5/100, 5/100, 1/100, 1/100, 5/10, 5/10]
#REWARD_WEIGHTS = [1/100, 1/100, 1/100, 1/100, 1/10, 1/10]

SUCCESS_THRESHOLD = 0.15

def distance_2_goal_reward(achieved_goal, desired_goal, p=0.5):
    return - np.power(np.dot(OBS_SCALE * np.abs(achieved_goal - desired_goal), REWARD_WEIGHTS), p)

if __name__ == "__main__":
    # lets suppose an error of 10 cm in x & y, 0.5m/s error and about 5 degrees = 0.0872665    
    err_angle = np.deg2rad(7)

    achieved_goal = np.array([0.10, 0.10, 0.10, 0.10, np.cos(err_angle), np.sin(err_angle)])
    desired_goal  = np.array([0.0, 0.0, 0.0, 0.0, np.cos(0), np.sin(0)])
    print (distance_2_goal_reward(achieved_goal, desired_goal))
