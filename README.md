# rl_urban_baselines

### Introduction

Reinforcement Learning Baselines (OpenAI) applied to Autonomous Driving

This Research is aiming to address RL approaches to solve 3 Urban scenarios: Roundabout, Merging and Urban/Street navigation.

The objective is to compare performances of well known RL algorithms (Open AI baselines) in these environments.

The research will also address the differences (advantages and disadvantages) between Continuous and Discrete Action Spaces. 

The Original version of these environments were created by Edouard Leurent and can be found in https://github.com/eleurent/highway-env

### Getting Started

1. Clone/Download this repository

2. Install all the required libraries: pip install -r requirements.txt

3. Install the urban_AD_env: 

   cd urban_AD_env

   python setup.py install

4. On main.py  "create_args()" function

   1. Select what algorithm from Openai baselines (alg='---') you would like to use (take into account that some will fail since they are meant for discrete action spaces and at this point we are not automatically switching from Continuous to Discrete action spaces). For a list of the algorithms available look into the /baselines folder.

   2. You may also select the "network" type (mlp, cnn, lstm, cnn_lstm, conv_only) otherwise Openai baselines will use a default one for the specific algorithm. 

   3. Look at the other arguments and chose the appropriate values such us where to save, number steps,...

   4. Note: "Save path" does not work on DDPG unless you add to the ddpg class the following: 
   
      from baselines.common import tf_util

      def save(self, save_path):

      ​        tf_util.save_variables(save_path)

      "Play" will launch the visualization and run the agent after training. (comment this line if you don't want to run the agent after training or if you can't visualize it - such as when you are training the agent on AWS, for example)

5. Run main_gym.py:

   python main_gym.py