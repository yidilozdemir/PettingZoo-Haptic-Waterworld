## Haptic- Waterworld

This is an extension of Pettingzoo's Waterworld environment, where agents are embodied with arousal and satiety, and can modulate these parameters via eating and social contact through haptic touch/proximity. The idea is to investigate the effects of arousal modulation via social contact in a cooperative resource sharing behaviour. 


The original Waterworld environment: https://pettingzoo.farama.org/environments/sisl/waterworld/


The Waterworld environment contains "pursuers", agents, whose primary goal is to eat food sources "evaders", and cooperation is implicitly expected as agents can only eat food if they are in physical proximity to each other. Pursuers have sensors to detect the moving objects near them and identify if they are other pursuers/ evader-food or poison; observational space of each agent. Waterworld has a contunious action space; where agents are controlled with a two dimensional velocity vector, by a policy that trains on the state of the environment and rewards recieved after each movement command is applied. I have built on this environment with the following concepts applied to the agents, and a new reward structure to encourage agents to modulate these concepts:


### Descriptions	

1) Satiety: Agents have a satiety parameter, that is initialised with a satiety value of 0.5. This gets modulated in environment with eating; there is a decay rate so decreases in natural conditions. No clipping     but a max_satiety value makes sure agents cannot eat more 
2) Arousal: Agents have an arousal parameter, that is initialised at 0. This gets modulated in environment with eating as well as social contact; where agents stand physically next to each other. There is a          decay rate so decreases in natural conditions. Gets clipped to [-1,1]
3) Eating: If more than 2 agents, whose satiety values are below max_satiety, are near the food, eating happens. Increases satiety and slightly increases arousal.
4) Social contact: When agents are together, in average mode, their arousal levels get pulled to the "group" average


#### Adjusted reward structure

- Control reward: Inherited from Waterworld, this penalises large thursts, to encourage smoother/no big changes
- Behaviour reward: Inherited from Waterworld, food eating reward (biggest) + small poison encounter penalty + "shaping reward": small reward from agents touching each other when there is no eating
- Satiety reward: Small reward for satiety values; Since satiety is capped in the system (max_satiety) there is no penalty for too high 
- Arousal penalty: To discourage too high or too low arousal levels


## Chosen Multi-agent Reinforcement Learning (MARL) Algorithm 

I chose to use the Proximal Policy Optimisation (PPO) learning to train a Multi-layer Peceptron(Mlp) to control the agents in a "centralised learning, decentralised execution" approach. Mlp is recommended to use in a continuious action space, as suggested by the SB3 library training tutorial designed for Waterworld https://pettingzoo.farama.org/tutorials/sb3/waterworld/


In training, a central PPO policy is trained, and during execution, each agent interacts with the policy with their own observation and action space; this type of parameter sharing policy approach, based on conversations with colleagues and online material, is a good enough approach with PPO to train multi agents. This is evidenced with the fact that many centralised learning, decentralised execution learning approaches like MADDPG are popular in MARL.
" Also look into MAPPO which utilises shared observations in the value function for improved cooperation. I generally use one of these for everything. They are not necessarily state-of-the-art but in my personal opinion they are nearly always the best way to go."
https://github.com/DLR-RM/stable-baselines3/issues/1817


I am using SB3 library, and my training is modified from the SB3 tutorial from PettingZoo for Waterworld. In the training, I incorporated an EvalCallback to periodically evaluate the trained model to pick the best one, and a general Callback function to plot rewards and social contact over time. 

## Getting startedL: Python environment 

I am using a conda environment with Python 3.11, pettingzoo and sb3 installed via pip with their dependencies. 

## Use

Currently, both training and evaluation code is in the same script in folder `sb3-training`

## Citation

To cite this project in publication, please use

```
@article{terry2021pettingzoo,
  title={Pettingzoo: Gym for multi-agent reinforcement learning},
  author={Terry, J and Black, Benjamin and Grammel, Nathaniel and Jayakumar, Mario and Hari, Ananth and Sullivan, Ryan and Santos, Luis S and Dieffendahl, Clemens and Horsch, Caroline and Perez-Vicente, Rodrigo and others},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={15032--15043},
  year={2021}
}
```
