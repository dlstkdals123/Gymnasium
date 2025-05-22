import gymnasium as gym
from Q4_policy_evaluation import PolicyIteration

env = gym.make("Taxi-v3", render_mode="human")
env = env.unwrapped

# Init
state, info = env.reset(seed = 1)
iterations = 30
# episodes = 10
delta = 0.00001
discount_factor = 0.99
policyIteration = PolicyIteration(env)

# 반복 시작
for iteration in range(iterations):
    env.render()

    policyIteration.policy_evaluation(delta = delta, discount_factor=discount_factor)
    policyIteration.policy_improvement(discount_factor=discount_factor)

    policyIteration.print_value(state)

    policy = policyIteration.get_policy(state, debug=True)
    action = policyIteration.get_action(state, debug=True)
    # policyIteration.print_board(state, info['action_mask'])

    state, reward, terminated, truncated, info = env.step(action)

    if terminated:
        print("Finish")
        print("total iterations:", iteration)
        break
    print("-" * 20)
env.close()