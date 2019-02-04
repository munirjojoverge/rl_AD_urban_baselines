######################################################################
#          Deep Reinforcement Learning for Autonomous Driving
#                  Created/Modified on: January 10, 2019
#                      Author: Munir Jojo-Verge
#######################################################################

import sys, os
from os.path import dirname, abspath
import datetime

file_path = sys.argv[0]
pathname = os.path.dirname(file_path)        
open_ai_baselines_dir = pathname + '/open_ai_baselines'
sys.path.append(open_ai_baselines_dir)

import baselines.run as run

from settings import *

import urban_AD_env

def create_args():
    """
    Create an argparse
    """    
    currentDT = datetime.datetime.now()
    env = 'urban-roundabout-v0'
    alg = 'her'
    network = 'default'

    # define the name of the directory to be created
    save_path = model_folder + '/' + alg

    try:  
        os.mkdir(save_path)
    except OSError:  
        print ("Creation of the save path %s failed. It might already exist" % save_path)
    else:  
        print ("Successfully created the save path folder %s " % save_path)

    args = []
    args.append('--env=' + env)  # environment ID  
    args.append('--alg='+ alg) # Algorithm
    args.append('--num_timesteps=1e5')
    args.append('--num_env=0') # Number of environment copies being run in parallel. When not specified, set to number of cpus for Atari, and to 1 for Mujoco
    args.append('--reward_scale=1.0') # Reward scale factor. Default: 1.0
    args.append('--save_path='+ save_path +'/'+ network + '_' + str(currentDT)) # Path to save trained model to
    args.append('--load_path='+ save_path +'/'+ network + '_2019-02-01 18:22:59.659116') # Path to save trained model to
    #args.append('--save_video_interval = 0') #Save video every x steps (0 = disabled)
    #args.append('--save_video_length = 200') # Length of recorded video. Default: 200
    #args.append('--network=' + network) # help='network type (mlp, cnn, lstm, cnn_lstm, conv_only)', default=None)
    args.append('--play') # Play after trainning
    # args.append('--extra_import = urban_AD_env') # Extra module to import to access external environments'


    return args

def create_dirs():
    for dirName in req_dirs:
        if not os.path.exists(dirName):
            os.mkdir(dirName)
            print("Directory " , dirName ,  " Created ")
        else:    
            print("Directory " , dirName ,  " already exists")

if __name__ == '__main__':
    create_dirs()
    args = create_args()
    run.main(args)