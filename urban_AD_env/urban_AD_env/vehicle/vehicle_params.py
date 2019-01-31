######################################################################
#          Deep Reinforcement Learning for Autonomous Driving
#                  Created/Modified on: January 10, 2019
#                      Author: Munir Jojo-Verge
#######################################################################

import math

SEED = 777
#### MTCS CONFIG
SEED = 7777
dt = 0.5	# MTCS time step (secs) for every action taken
HORIZON = 3.0 # Desired Simulation/prediction Horizon in Seconds
BEHAVIOR_PLANNER_dt = 1.0 # We will "look" at the diving scene every "this time" and plan for the next "HORIZON"
MTCS_DEPTH = int(HORIZON / dt)	# ex: 3sec/0.5 = 6
NUM_ACTIONS_TO_SAMPLE = 2
NUM_ACTIONS_STEPS = 5  # Since our action space represents "Changes" (in steering, i.e steering rate, and acceleration, i.e jerk, 
                       # we allow "NUM_ACTIONS_STEPS" to the left and the same to the right from the actual state.
					   # For example we might be accelerating at 1m/s2 and the recommended action might be:
					   # to stay, i.e 0 steps change from where we are.
					   # to deccelerate 2 steps (the sice of the step is determined by the dt used)
					   # or to accle 1 step.
					   # In the case of "NUM_ACTIONS_STEPS = 10" means that we have 21 possible actions: -10, -9,....0, 1, 2,...10. (Step increments)

### VEHICLE PARAMETERS
# Steering angle Max Change per step time allowed: Calculation
MAX_STEER_WHEEL_ANGLE_RATE = 6.981317008 #RADS/S   #400.00 #DEGREES/S    ===> THIS IS HARDWARE SPECIFIC. THE POWER STEERING CAN NOT GO FASTER
STEERING_WHEEL_TO_TIRE_RATIO = 16
MAX_STEER_ANGLE_RATE = MAX_STEER_WHEEL_ANGLE_RATE / STEERING_WHEEL_TO_TIRE_RATIO
MAX_STEER_WHEEL_ANGLE = 9.42478 # Rads = 540 Degrees
MAX_STEER_ANGLE = MAX_STEER_WHEEL_ANGLE / STEERING_WHEEL_TO_TIRE_RATIO

# Define Steering angle rates regions according to "DRIVING STYLES". For now just divide it 3 equal regions
# MAX_STEER_ANGLE_RATE => MSRA 
MSAR_GENTLE =  MAX_STEER_ANGLE_RATE / 3
MSAR_NORMAL =  MSAR_GENTLE * 2  
MSAR_SPORTY =  MSAR_GENTLE * 3

#### THIS IS WHAT WILL USE TO GENERATE THE "CHANGE" COMMAND (MULTIPLIED BY OUR dt TO GET AN ANGLE CHANGE)
MSAR = [MSAR_GENTLE, MSAR_NORMAL, MSAR_SPORTY]

# STEERING RESOLUTION: MIN CHANGE IN ANGLE THAT THE POWER STEERING CAN PERFORM (NOT SURE WHAT IS THE MIN TIME STEP THAT IT CAN BE ACHIEVED)
STEERING_RESOLUTION = 0.00174533 # Radians, 0.1 DEGREES


# Acceleration Max Change per step time allowed
MAX_ACCEL = 2.94 # ==> VEHICLE/HARDWARE
MAX_DECEL = -6.00

# From: Determination of Minimum Horizontal Curve Radius Used in the Design 
#       of Transportation Structures, Depending on the Limit Value of Comfort
#       Criterion Lateral Jerk
#       Ahmet Sami KILINÇ and Tamer BAYBURA, Turkey
MAX_LAT_JERK_GENTLE = 0.3 # m/s^2. 0.0 <= max lat jerk < 0.3
MAX_LAT_JERK_NORMAL = 0.9 # m/s^2  0.3 <= max lat jerk < 0.9
MAX_LAT_JERK_SPORTY = 1.5 # m/s^2  0.9 <= max lat jerk < 1.5

MAX_LON_JERK_GENTLE = 0.5 # m/s^2. 0.0 <= max lon jerk < 0.5
MAX_LON_JERK_NORMAL = 1.1 # m/s^2  0.3 <= max lon jerk < 0.9
MAX_LON_JERK_SPORTY = 2.0 # m/s^2  0.9 <= max lon jerk < 1.5

# For simplicity we'll work with jerk magnitude
MAX_JERK_GENTLE = math.sqrt(MAX_LAT_JERK_GENTLE**2 + MAX_LON_JERK_GENTLE**2)
MAX_JERK_NORMAL = math.sqrt(MAX_LAT_JERK_NORMAL**2 + MAX_LON_JERK_NORMAL**2)
MAX_JERK_SPORTY = math.sqrt(MAX_LAT_JERK_SPORTY**2 + MAX_LON_JERK_SPORTY**2)

#### THIS IS WHAT WILL USE TO GENERATE THE "CHANGE" COMMAND (MULTIPLIED BY OUR dt TO GET AN ANGLE CHANGE)
MAX_JERK = [MAX_JERK_GENTLE, MAX_JERK_NORMAL, MAX_JERK_SPORTY]



NUM_EPISODES = int(3e3)     # Max number of episodes
T_MAX        = int(1e3)     # Max Number of training steps (This is only in case within an Episode you never get it "done" - Never crashes, gets out the road, etc.. Which only shows a failure on the SIM)
MCTS_NUM_SIMS = int(NUM_ACTIONS_TO_SAMPLE**MTCS_DEPTH)
MEMORY_SIZE = int(1e6)      # Experience replay memory capacity
TURNS_UNTIL_TAU0 = 10 		# turn on which iteration starts playing deterministically
CPUCT = 1
EPSILON = 0.25				# Additional exploration is achieved by adding Dirichlet noise to the
							# prior probabilities in the root node s 0 , specifically P (s, a) = (1 − eps)p a + eps*ηa , where η ∼ Dir(0.03)
							# and eps = 0.25; this noise ensures that all moves may be tried, but the search may still overrule bad moves

ALPHA = 0.8 				# For Dirichlet Noise: positive real number called the concentration parameter (also known as scaling parameter)


#### RETRAINING
BATCH_SIZE = 256
EPOCHS = 1
REG_CONST = 0.0001
LEARNING_RATE = 1e-3
MOMENTUM = 0.9
TRAINING_LOOPS = 1
INPUT_SHAPE = (3,12) # but we will have n frames
INPUT_FRAMES = 3
ACTION_SHAPE = ((2 * NUM_ACTIONS_STEPS) + 1) ** 2 # table steering changes Vs Accel Changes - That's the reason you see a squared (**2)

LEARNING_STARTS_RATIO = 1/500  # Number of steps before starting training = memory capacity * this ratio
LEARNING_FREQUENCY = 20        # Steps before we sample from the replay Memory again    

reward_clip = 0.5               # Reward clipping (0 to disable)



HIDDEN_CNN_LAYERS = [
	{'filters':75, 'kernel_size': (4,4)}
	 , {'filters':75, 'kernel_size': (4,4)}
	 , {'filters':75, 'kernel_size': (4,4)}
	 , {'filters':75, 'kernel_size': (4,4)}
	 , {'filters':75, 'kernel_size': (4,4)}
	 , {'filters':75, 'kernel_size': (4,4)}
	]

#### EVALUATION
EVAL_NUM_EPISODES = 20
SCORING_THRESHOLD = 1.3