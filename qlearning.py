import numpy as np
import random

import gym

env = gym.make("Taxi-v2").env
env.render()


q_table = np.zeros([env.observation_space.n, env.action_space.n])

# 5 x 5 -> Grid
# 4 -> pick up/drop off locations
# 4 + 1 -> passenger locations
# 5 x 5 x 4 x (4 + 1)
print("States: " + str(env.observation_space.n))


# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1

# for plotting metrics
all_epochs = []
all_penalties = []

for i in range(1, 100001):
    state = env.reset()

    epochs, penalties, reward, = 0, 0, 0
    done = False

    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() # explore action space
        else:
            action = np.argmax(q_table[state]) # exploit learned values

        next_state, reward, done, info = env.step(action)

        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])

        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        if reward == -10:   # 10 point penalty for illegal pick-up and drop-off actions
            penalties += 1  # -1 for every wall hit

        state = next_state
        epochs += 1

    if i % 100 == 0:
        #clear_output(wait=True)
        print("Episode: " + str(i))

print("Training finished.")

# Print the Q-Table.
print(q_table)

print("-----------------------------------------------------------------------")
print("Evaluate the model.")
print("-----------------------------------------------------------------------")

# Evaluate agent's performance after Q-learning
total_epochs, total_penalties = 0, 0
episodes = 10

for episode in range(episodes):
    print("==============================================")
    print("Episode: " + str(episode))
    print("==============================================")

    state = env.reset()
    epochs, penalties, reward = 0, 0, 0

    done = False

    while not done:
        action = np.argmax(q_table[state])
        state, reward, done, info = env.step(action)

        env.render()

        if reward == -10:
            penalties += 1

        epochs += 1

    total_penalties += penalties
    total_epochs += epochs

print("Results after " + str(episodes) + " episodes.")
print("Average timesteps per episode: " + str(total_epochs / episodes))
print("Average penalties per episode: " + str(total_penalties / episodes))
