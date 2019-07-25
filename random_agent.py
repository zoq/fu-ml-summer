import numpy as np
import random
import gym

#Create environment (https://github.com/openai/gym/wiki/CartPole-v0)

## Action
# 0 - Push cart to the left
# 1 - Push cart to the right

##Observation
# - Cart Position   -2.4    2.4
# - Cart Velocity   -Inf    Inf
# - Pole Angle  -41.8°    +41.8°
# - Pole Velocity At Tip    -Inf    Inf

## Episode Termination
# - Pole Angle is more than +/-12°
# - Cart Position is more than +/-2.4 (center of the cart reaches the edge of the display)
# - Episode length is greater than 200

env = gym.make('CartPole-v0')
for i_episode in range(200):
    observation = env.reset()
    rewards = 0

    for t in range(250):
        env.render()

        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        rewards += reward
        if done:
            print('Episode finished after {} timesteps, total rewards {}'.format(t + 1, rewards))
            break

env.close()