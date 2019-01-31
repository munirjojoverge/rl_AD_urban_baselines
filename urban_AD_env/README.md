# urban-env

A collection of environments for *urban driving* and tactical decision-making tasks

<p align="center">
    <img src="docs/media/urban-env.gif"><br/>
    <em>An episode of one of the environments available in higwhay-env.</em>
</p>

[![Build Status](https://travis-ci.org/eleurent/urban-env.svg?branch=master)](https://travis-ci.org/eleurent/urban-env)
[![Docs Status](https://img.shields.io/badge/docs-passing-green.svg)](https://eleurent.github.io/urban-env/)

## Installation

`pip install --user git+https://github.com/eleurent/urban-env`

## Usage

```python
import higwhay_env

env = gym.make("urban-v0")

done = False
while not done:
    action = ... # Your agent code here
    obs, reward, done, _ = env.step(action)
    env.render()
```

## The environments

### urban

```python
env = gym.make("urban-v0")
```

In this task, the ego-vehicle is driving on a multilane urban populated with other vehicles.
The agent's objective is to reach a high velocity while avoiding collisions with neighbouring vehicles. Driving on the right side of the road is also rewarded.

<p align="center">
    <img src="docs/media/urban.gif"><br/>
    <em>The higwhay-v0 environment.</em>
</p>


### Merge

```python
env = gym.make("urban-merge-v0")
```

In this task, the ego-vehicle starts on a main urban but soon approaches a road junction with incoming vehicles on the access ramp. The agent's objective is now to maintain a high velocity while making room for the vehicles so that they can safely merge in the traffic.

<p align="center">
    <img src="docs/media/merge.gif"><br/>
    <em>The urban-merge-v0 environment.</em>
</p>

### Roundabout

```python
env = gym.make("urban-roundabout-v0")
```

In this task, the ego-vehicle if approaching a roundabout with flowing traffic. It will follow its planned route automatically, but has to handle lane changes and longitudinal control to pass the roundabout as fast as possible while avoiding collisions.

<p align="center">
    <img src="docs/media/roundabout-env.gif"><br/>
    <em>The urban-roundabout-v0 environment.</em>
</p>

## The framework

New urban driving environments can easily be made from a set of building blocks.

### Roads

A `Road` is composed of a `RoadNetwork` and a list of `Vehicles`. The `RoadNetwork` describes the topology of the road infrastructure as a graph, where edges represent lanes and nodes represent intersections. For every edge, the corresponding lane geometry is stored in a `Lane` object as a parametrized center line curve, providing a local coordinate system.

### Vehicle kinematics

The vehicles kinematics are represented in the `Vehicle` class by a _Kinematic Bicycle Model_.

<a href="https://www.codecogs.com/eqnedit.php?latex=\dot&space;x=v\cos\psi" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dot&space;x=v\cos\psi" title="\dot x=v\cos\psi" /></a><br>
<a href="https://www.codecogs.com/eqnedit.php?latex=\dot&space;y=v\sin\psi" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dot&space;y=v\sin\psi" title="\dot y=v\sin\psi" /></a><br>
<a href="https://www.codecogs.com/eqnedit.php?latex=\dot&space;v=a" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dot&space;v=a" title="\dot v=a" /></a><br>
<a href="https://www.codecogs.com/eqnedit.php?latex=\dot\psi=\frac{v}{l}\tan\beta" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dot\psi=\frac{v}{l}\tan\beta" title="\dot\psi=\frac{v}{l}\tan\beta" /></a>

Where *(x, y)* is the vehicle position, *v* its forward velocity and *psi* its heading.
*a* is the acceleration command and *beta* is the slip angle at the center of gravity, used as a steering command.

### Control

The `ControlledVehicle` class implements a low-level controller on top of a `Vehicle`, allowing to track a given target velocity and follow a target lane.

### Behaviours

The vehicles populating the urban follow simple and realistic behaviours that dictate how they accelerate and steer on the road.

In the `IDMVehicle` class,
* Longitudinal Model: the acceleration of the vehicle is given by the Intelligent Driver Model (IDM) from [(Treiber et al, 2000)](https://arxiv.org/abs/cond-mat/0002177).
* Lateral Model: the discrete lane change decisions are given by the MOBIL model from [(Kesting et al, 2007)](https://www.researchgate.net/publication/239439179_General_Lane-Changing_Model_MOBIL_for_Car-Following_Models).

In the `LinearVehicle` class, the longitudinal and lateral behaviours are defined as linear weightings of several features, such as the distance and velocity difference to the leading vehicle.

## The agents

Agents solving the `urban-env` environments are available in the [RL-Agents](https://github.com/eleurent/rl-agents) repository.

`pip install --user git+https://github.com/eleurent/rl-agents`

### [Deep Q-Network](https://github.com/eleurent/rl-agents/tree/master/rl_agents/agents/dqn)


<p align="center">
    <img src="docs/media/dqn.gif"><br/>
    <em>The DQN agent solving urban-v0.</em>
</p>

This model-free reinforcement learning agent performs Q-learning with function approximation, using a neural network to represent the state-action value function Q.


### [Value Iteration](https://github.com/eleurent/rl-agents/blob/master/rl_agents/agents/dynamic_programming/value_iteration.py)

<p align="center">
    <img src="docs/media/ttcvi.gif"><br/>
    <em>The Value Iteration agent solving urban-v0.</em>
</p>

The Value Iteration is only compatible with finite discrete MDPs, so the environment is first approximated by a [finite-mdp environment](https://github.com/eleurent/finite-mdp) using `env.to_finite_mdp()`. This simplified state representation describes the nearby traffic in terms of predicted Time-To-Collision (TTC) on each lane of the road. The transition model is simplistic and assumes that each vehicle will keep driving at a constant velocity without changing lanes. This model bias can be a source of mistakes.

The agent then performs a Value Iteration to compute the corresponding optimal state-value function.


### [Monte-Carlo Tree Search](https://github.com/eleurent/rl-agents/blob/master/rl_agents/agents/tree_search/mcts.py)

This agent leverages a transition and reward models to perform a stochastic tree search [(Coulom, 2006)](https://hal.inria.fr/inria-00116992/document) of the optimal trajectory. No particular assumption is required on the state representation or transition model.

<p align="center">
    <img src="docs/media/mcts.gif"><br/>
    <em>The MCTS agent solving urban-v0.</em>
</p>
