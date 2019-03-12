######################################################################
#          Deep Reinforcement Learning for Autonomous Driving
#                  Created/Modified on: February 5, 2019
#                      Author: Munir Jojo-Verge
#######################################################################

import sys, os
from os.path import dirname, abspath
import time

file_path = sys.argv[0]
pathname = os.path.dirname(file_path) 

open_ai_baselines_dir = pathname + '/open_ai_baselines'
print(open_ai_baselines_dir)

urban_AD_env_path = pathname + '/urban_AD_env/envs'
print(urban_AD_env_path)

sys.path.append(open_ai_baselines_dir)
sys.path.append(urban_AD_env_path)

import baselines.run as run

import urban_AD_env

from settings import req_dirs, models_folder

def create_dirs(req_dirs):
    for dirName in req_dirs:
        if not os.path.exists(dirName):
            os.mkdir(dirName)
            print("Directory " , dirName ,  " Created ")
        else:    
            print("Directory " , dirName ,  " already exists")


def default_args():    
    create_dirs(req_dirs)

    currentDT = time.strftime("%Y%m%d-%H%M%S")
    ###############################################################
    #            DEFINE YOUR "BASELINES" PARAMETERS HERE 
    ###############################################################
    env =  'parking-v1' #'urban_AD-multilane-v1' #'sidepass-v0' #'urban_AD-merge-v1' #'parking-v1' #'continuous-multi-env-v1' #'continuous-env-v1' 'parking-v1'
    alg = 'her'
    network = 'default'
    num_timesteps = '3e4'
    save_folder = models_folder + '/' + env +'/'+ alg + '/' + network 
    save_file = save_folder + '/' + str(currentDT)
    logger_path = save_file + '_log'
    load_path = save_folder +'/'+ '20190228-174333' #her_default_20190212-141935' # Good with just Ego
    # load_path = save_folder +'/'+ 'her_default_obs5_20190212-202901' # So-So with others
    ###############################################################
        
    try:  
        os.mkdir(save_folder)
    except OSError:  
        print ("Creation of the save path %s failed. It might already exist" % save_folder)
    else:  
        print ("Successfully created the save path folder %s " % save_folder)

    DEFAULT_ARGUMENTS = [
        '--env=' + env,
        '--alg=' + alg,
    #    '--network=' + network,
        '--num_timesteps=' + num_timesteps,    
    #    '--num_env=0',
        '--save_path=' + save_file,
        '--load_path=' + load_path,
        '--logger_path=' + logger_path,
        '--play'
    ]

    return DEFAULT_ARGUMENTS


if __name__ == "__main__":
    args = sys.argv
    if len(args) <= 1:
        args = default_args()
    run.main(args)
