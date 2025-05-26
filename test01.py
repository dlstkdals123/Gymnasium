import gymnasium as gym
from test01_policy_evaluation import PolicyIteration

env = gym.make("Taxi-v3", render_mode="human")
env = env.unwrapped

# Init
state, info = env.reset(seed = 1)
iterations = 0
episodes = 0
max_action_numbers = 30
discount_factor = 0.99
delta = 1
policyIteration = PolicyIteration(env)

# 반복 시작
policy_stable = False
while not policy_stable:
    iterations += 1
    policyIteration.Q4_policy_evaluation(delta = delta, discount_factor=discount_factor)
    policy_stable = policyIteration.Q4_policy_improvement(discount_factor=discount_factor)

print("===================")
print(f"Total iterations: {iterations}")
print("===================")
policyIteration.print_board(state, info["action_mask"])
    
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