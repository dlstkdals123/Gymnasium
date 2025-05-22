import gymnasium as gym
from test01_policy_evaluation import PolicyIteration
import matplotlib.pyplot as plt

def init():
    env = gym.make("Taxi-v3", render_mode="human")
    env = env.unwrapped
    global state
    state, _ = env.reset(seed = 1)
    return PolicyIteration(env)

def policy_iteration_start(policy_iteration):
    # Init
    iterations = 0
    discount_factor = 0.99

    # 반복 시작
    policy_stable = False
    while not policy_stable:
        iterations += 1
        policy_iteration.Q4_policy_evaluation(delta = 0.00001, discount_factor=discount_factor)
        policy_stable = policy_iteration.Q4_policy_improvement(discount_factor=discount_factor)

    print("===================")
    print(f"Total iterations: {iterations}")
    print("===================")

def simulation(policy_iteration):
    max_action_numbers = 30
    global state
    for action_number in range(max_action_numbers):
        print(f"action number: {action_number + 1}")
        policy_iteration.print_value(state)
        policy = policy_iteration.get_policy(state, debug=True)
        action = policy_iteration.get_action(state, debug=True)

        state, reward, terminated, truncated, info = policy_iteration.env.step(action)

        if terminated:
            break
        print("-" * 20)

if __name__ == "__main__":
    policy_iteration = init()
    policy_iteration_start(policy_iteration)
    simulation(policy_iteration)