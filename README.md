## Haptic- Waterworld

This is an extension of Pettingzoo's Waterworld environment, where agents are embodied with arousal and satiety, and can modulate these parameters via eating and social contact through haptic touch/proximity. The idea is to investigate the effects of arousal modulation via social contact in a cooperative resource sharing paradigm.


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

#### Relationship between concepts and emerging behaviour

##### Satiety and arousal:

When satiety is low (high hunger), it causes a gradual increase in arousal over time. 
When arousal is high, it causes a faster decay rate in satiety resulting in lower satiety.

This creates an interesting dynamic:
Being hungry (low satiety) slowly increases arousal over time.
Having more arousal makes the agent more hungry over time. 

We currently do not have an arousal boost from eating, to keep the system relatively straightforward to study. For future work in this direction, that is kept as a variable in system `satiety_arousal_rate` which is set to 0. That case could create a more realistic scenario where this system models both the immediate stimulation from eating and the growing agitation from prolonged hunger. The balance between these effects, along with the social modulation and other factors, contributes to the complex behavior of the agents in your simulation.

Here, the cyclical relationship between satiety and arousal is expected to continue driving the food seeking behaviour. 

#####  Eating and satiety:

Eating increases satiety if the agent's satiety is below the maximum value. 

#####  Eating and arousal:

Eating is a social activity in this simulation. It requires multiple pursuers to be in close proximity to the food source, thus the arousal gets affected in two main ways; a) The act of eating (increasing satiety which creates a faster decay in arousal over time) b) The social contact with other pursuers (through haptic modulation)

This creates a more complex dynamic where:
Hungry pursuers need to not only find food but also coordinate with other pursuers to actually consume it.
The act of eating inherently involves social interaction, which affects arousal through both food consumption and social modulation.
The arousal changes from eating and social contact happen at the same time, but the change coming from the eating behaviour is more spread over time, potentially amplifying or dampening each other's effects.

#####  Social contact and arousal-satiety: 

Social contact pulls the arousal of the agents in contact to the average of the group (average mode) In this scenario, after a contact, based on whether agent's arousal rate got higher or lower, different behaviours can emerge: 
1. Agent's arousal rate increases after contact during eating -> high satiety from eating but fast decay -> arousal overall gets higher, which is penalising -> to decrease/modulate arousal social contact can happen or food might be preferred, entering a fast satiety decay/food eating cyclic behaviour
2. Agent's arousal rate decreases after contact -> high satiety from food and can be sustained more due to low arousal, which is also rewarding -> which can encourage less eating 

### Analysis Strategy
Dynamics become harder to reason, and we will test across conditions to pick them apart:

1)  No arousal modulation vs arousal modulation only in no-food contact vs arousal modulation in food contact
2)  Training with arousal modulation vs training with no arousal modulation -> then testing with no arousal modulation to see if this encouraged more or less cooperative behaviour 

##### Social eating mechanic adds another layer of complexity to the system, potentially leading to interesting emergent behaviors like:

Pursuers clustering around food sources
Coordination or competition between pursuers for eating opportunities
Complex arousal dynamics due to the combination of eating and social effects

#### Metrics
- visual observation
- average rewards, across and between agents
- Median rewards across eval runs + standard deviation
- Tracking satiety and arousal before/after episode, and standard deviation throughout episode as a marker for how stable it was for each agent

Plotting ideas in eval and training round
- physical contact across episode
- food eating across episode
- 

#### The code for this environment

1)  Models: define Pursuers agent and other "objects" in the environment
https://github.com/yidilozdemir/PettingZoo-Haptic-Waterworld/blob/main/pettingzoo/sisl/waterworld/waterworld_model1_models.py


```
        self.satiety = initial_satiety
        self.arousal = initial_arousal
        self.max_satiety = max_satiety
        self.satiety_decay_rate = satiety_decay_rate
        self.arousal_decay_rate = arousal_decay_rate
        self.haptic_modulation_type = haptic_modulation_type
        self.satiety_arousal_rate = satiety_arousal_rate
        
```

Logic that updates these variables in each action step is inside the `update` function

```
def update(self, dt, other_pursuers, evaders):

        # Update position
        self.body.position += self.body.velocity * dt

        # Calculate arousal based on nearby pursuers and social haptic modulation due to these contacts
        self.social_haptic_modulation = 0
        pursuers_in_contact = [self]
        total_arousal = self.arousal
        for other in other_pursuers:
            if other != self and self.distance_to(other) < self.radius + other.radius:
                pursuers_in_contact.append(other)
                total_arousal += other.arousal
                max_arousal = max(max_arousal, other.arousal)
                min_arousal = min(min_arousal, other.arousal)
        if len(pursuers_in_contact) > 1:
            if self.haptic_modulation_type == "average":
                average_arousal = total_arousal / len(pursuers_in_contact)
                self.social_haptic_modulation = (average_arousal - self.arousal) * 0.1
            elif self.haptic_modulation_type == "cooperative":
                self.social_haptic_modulation = (max_arousal - self.arousal) * 0.1
            elif self.haptic_modulation_type == "competitive":
                self.social_haptic_modulation = (min_arousal - self.arousal) * 0.1
            else:
                self.social_haptic_modulation = 0
        else:
            self.social_haptic_modulation = 0


        # Update food awareness
        awareness_range = self.sensor_range * (1 + self.arousal)
        self.food_awareness = [
            evader for evader in evaders
            if self.distance_to(evader) < awareness_range and not evader.eaten
        ]

        #Update arouusal influenced both by hunger and arousal modulation coming from haptic contact
        hunger = 1 - self.satiety
        self.arousal += hunger * 0.01 * dt + self.social_haptic_modulation
        self.arousal = np.clip(self.arousal, -1, 1)

         # Update satiety 
        satiety_decrease_rate = 0.05 * (1 + self.arousal)
        self.satiety = max(self.satiety - satiety_decrease_rate * dt, 0)
        #radius doesnt change on satiety but TODO for arousal increasing sensor change 
        #self.radius = max(self.satiety * 10, 1)  # Ensure a minimum radius
        #self.shape.unsafe_set_radius(self.radius)
        
        # Decay and clip arousal
        if self.arousal > 0:
            self.arousal = max(0, self.arousal - self.arousal_decay_rate * dt)
        else:
            self.arousal = min(0, self.arousal + self.arousal_decay_rate * dt)
        
        #clip to be in certain range
        self.arousal = np.clip(self.arousal, -1, 1)
```
Eating behaviour

```
def eat(self, food_nutrition):
        previous_satiety = self.satiety
        self.satiety = min(1, self.satiety + food_nutrition)
        
        # Eating success impacts arousal
        self.arousal += self.satiety_arousal_rate  # Set to 0 so no change in arousal from eating behaviour
        self.arousal = np.clip(self.arousal, -1, 1)

````


2) Base: the reward structure is inside the step method
https://github.com/yidilozdemir/PettingZoo-Haptic-Waterworld/blob/main/pettingzoo/sisl/waterworld/waterworld_model1_base.py

`step` function calculates the new velocity as defined by the policy, uses pymunk library to execute this in the physical plane, and calculates the new control and behaviour reward. Then it calls a seperate `calculate_rewards` function that combines these rewards with my custom satiety reward and arousal penalty, with a globally distributed score to encourage cooperation. 

```
def _calculate_rewards(self):
        self.rewards = [0 for _ in range(self.n_pursuers)]
        
        for i in range(self.n_pursuers):
            pursuer = self.pursuers[i]
            # Base reward from food, poison, etc.
            base_reward = self.behavior_rewards[i] + self.control_rewards[i]
            
            # Satiety reward: higher reward for maintaining high satiety
            satiety_reward = pursuer.satiety * self.satiety_reward_factor
            
            # Arousal penalty: penalty for very high or very low arousal
            arousal_penalty = -abs(pursuer.arousal) * self.arousal_penalty_factor
            
            # Combine rewards
            total_reward = base_reward + satiety_reward + arousal_penalty
            
            self.rewards[i] = total_reward

        # Calculate global reward
        global_reward = sum(self.rewards) / self.n_pursuers

        # Apply local vs global reward ratio
        for i in range(self.n_pursuers):
            self.rewards[i] = (
                self.rewards[i] * self.local_ratio
                + global_reward * (1 - self.local_ratio)
            )

    def step(self, action, agent_id, is_last):
        action = np.asarray(action) * self.pursuer_max_accel
        action = action.reshape(2)
        thrust = np.linalg.norm(action)
        if thrust > self.pursuer_max_accel:
            action = action * (self.pursuer_max_accel / thrust)

        p = self.pursuers[agent_id]

        # Update pursuer velocity
        _velocity = np.clip(
            p.body.velocity + action * self.pixel_scale,
            -p.max_speed,
            p.max_speed,
        )
        p.body.velocity = _velocity

        # Calculate thrust penalty (control reward)
        accel_penalty = self.thrust_penalty * math.sqrt((action**2).sum())
        self.control_rewards = (
            (accel_penalty / self.n_pursuers)
            * np.ones(self.n_pursuers)
            * (1 - self.local_ratio)
        )
        self.control_rewards[agent_id] += accel_penalty * self.local_ratio

        if is_last:
            # Update all pursuers
            for pursuer in self.pursuers:
                pursuer.update(1 / self.FPS, self.pursuers, self.evaders)

            self.space.step(1 / self.FPS)

            # Reset behavior rewards
            self.behavior_rewards = [0 for _ in range(self.n_pursuers)]

            # Handle food consumption and calculate behavior rewards
            for evader in self.evaders:
                pursuers_eating = [
                    pursuer for pursuer in self.pursuers
                    if pursuer.distance_to(evader) < pursuer.radius + evader.shape.radius
                ]
                total_satiety = sum(pursuer.satiety for pursuer in pursuers_eating)
                
                if total_satiety > evader.nutrition:
                    # Food is consumed
                    for pursuer in pursuers_eating:
                        pursuer.eat(evader.nutrition / len(pursuers_eating))
                        pursuer_index = self.pursuers.index(pursuer)
                        self.behavior_rewards[pursuer_index] += self.food_reward
                    evader.eaten = True
                else:
                    # Food is encountered but not consumed
                    for pursuer in pursuers_eating:
                        pursuer_index = self.pursuers.index(pursuer)
                        self.behavior_rewards[pursuer_index] += self.encounter_reward

            # Handle poison collisions
            for poison in self.poisons:
                for pursuer in self.pursuers:
                    if pursuer.distance_to(poison) < pursuer.radius + poison.shape.radius:
                        pursuer_index = self.pursuers.index(pursuer)
                        self.behavior_rewards[pursuer_index] += self.poison_reward
                        # Reset poison position
                        x, y = self._generate_coord(poison.shape.radius)
                        poison.body.position = x, y

            # Remove eaten evaders and spawn new ones
            self.evaders = [evader for evader in self.evaders if not evader.eaten]
            while len(self.evaders) < self.n_evaders:
                self.evaders.append(self._spawn_evader())

            # Calculate final rewards
            self._calculate_rewards()

            # Update frames
            self.frames += 1

        return self.observe(agent_id)
```

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
