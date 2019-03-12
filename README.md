# rl_urban_baselines

### Introduction

Reinforcement Learning Baselines (OpenAI) applied to Autonomous Driving

This Research is aiming to address RL approaches to solve 3 Urban scenarios: Roundabout, Merging and Urban/Street navigation.

The objective is to compare performances of well known RL algorithms (Open AI baselines) in these environments.

The research will also address the differences (advantages and disadvantages) between Continuous and Discrete Action Spaces. 

The Original version of these environments were created by Edouard Leurent and can be found in https://github.com/eleurent/highway-env

### Getting Started

Clone/Download this repository

Install all the required libraries: pip install -r requirements.txt

Install the urban_AD_env: 

cd urban_AD_env

**python setup.py install**

On **baslines_run.py**  you will find the "default_args()" function where you can setup the arguments that will be used to train and run the RL agent

1. env =  'urban_AD-v1' - **Chose the environment you want to use**. For a full list check "init__.py" on urban_AD_env folder
2. alg = 'ppo2' - **Chose the RL algorithm you want to use**. For a full list check the folders inside open_ai_baselines/baselines. Take into account that some will fail since they are meant for discrete action spaces and at this point we are not automatically switching from Continuous to Discrete action spaces
3. network = 'default' - **Chose the type of network you want to use**. Each algorithm can only handle certain network architectures. For more info you'll have to check inside the algorithm's folder you have selected. You can select  mlp, cnn, lstm, cnn_lstm, and conv_only otherwise Openai baselines will use a default one for the specific algorithm
4. num_timesteps = '3e5'
5. save_folder = models_folder + '/' + env +'/'+ alg + '/' + network 
6. save_file = save_folder + '/' + str(currentDT)
7. logger_path = save_file + '_log'
8. load_path = save_folder +'/'+ 'XXX' 
9. "Play" will launch render the environment and run the agent after training. (comment this line if you don't want to run the agent after training or if you can't visualize it - such as when you are training the agent on AWS, for example)

Look at the other arguments and chose the appropriate values such us where to save, number steps, etc...

Note: "Save path" does not work on DDPG unless you add to the ddpg class the following: 

from baselines.common import tf_util

def save(self, save_path):

​        tf_util.save_variables(save_path)

Comment the lines on DEFAULT_ARGUMENTS as shown below if you want to skip certain parameters

DEFAULT_ARGUMENTS = [

​        '--env=' + env,

​        '--alg=' + alg,

​    \#    '--network=' + network,

​        '--num_timesteps=' + num_timesteps,    

​    \#    '--num_env=0',

​        '--save_path=' + save_file,

​     /#   '--load_path=' + load_path,

​        '--logger_path=' + logger_path,

​        '--play'

​    ]

### How to Run

On the command prompt:

python baselines_run.py