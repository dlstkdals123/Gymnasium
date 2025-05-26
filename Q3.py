import gymnasium as gym
from policy_evaluation import PolicyIteration

env = gym.make("Taxi-v3", render_mode="human")
env = env.unwrapped

# Init
state, info = env.reset(seed = 1)
iterations = 10
episodes = 10
max_action_numbers = 30
discount_factor = 0.99
policyIteration = PolicyIteration(env)

# 반복 시작
for iteration in range(iterations):
    policyIteration.policy_evaluation(iter_num = episodes, discount_factor=discount_factor)
    policyIteration.policy_improvement(discount_factor=discount_factor)
    
for action_number in range(max_action_numbers):
    print(f"action number: {action_number + 1}")
    env.render()
    policyIteration.print_value(state)
    policy = policyIteration.get_policy(state, debug=True)
    action = policyIteration.get_action(state, debug=True)

    state, reward, terminated, truncated, info = env.step(action)

    if terminated:
        break
    print("-" * 20)
env.close()