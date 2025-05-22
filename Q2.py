import gymnasium as gym
import numpy as np
import time
from policy_evaluation import PolicyIteration

env = gym.make("Taxi-v3", render_mode="human")
env = env.unwrapped

# Init
state, info = env.reset()
iterations = 1
episodes = 100
discount_factor = 0.99
policyIteration = PolicyIteration(env)

# 반복 시작
for iteration in range(iterations):
    env.render()

    policyIteration.policy_evaluation(iter_num = episodes, discount_factor=discount_factor)
    policyIteration.policy_improvement(discount_factor=discount_factor)

    policyIteration.print_value(state)

    policy = policyIteration.get_policy(state)
    print(f"current state: {state}")
    print(f'policy: {policy}')
    action = policyIteration.get_action(state, debug=True)
    policyIteration.print_board(state, info['action_mask'])

    state, reward, terminated, truncated, info = env.step(action)

    if terminated:
        print("Finish")
        break
env.close()