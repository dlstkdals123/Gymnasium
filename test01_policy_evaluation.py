import gymnasium as gym
import numpy as np
from gymnasium import Env

np.random.seed(202211356)

# Constant
action_names = ["down", "up", "right", "left", "pickup", "dropoff"]

# 8방향 화살표 상수
ARROW_DOWN = "\u2193"  # 하 (↓)
ARROW_UP = "\u2191"  # 상 (↑)
ARROW_RIGHT = "\u2192"  # 우 (→)
ARROW_LEFT = "\u2190"  # 좌 (←)
ARROW_SOUTHEAST = "\u2198"  # 남동 (↘)
ARROW_SOUTHWEST = "\u2199"  # 남서 (↙)
ARROW_NORTHEAST = "\u2197"  # 북동 (↗)
ARROW_NORTHWEST = "\u2196"  # 북서 (↖)

# 4차원 배열을 None으로 초기화 (2x2x2x2)
arrow_array = [[[['X' for _ in range(2)] for _ in range(2)] for _ in range(2)] for _ in range(2)]

arrow_array[1][0][0][0] = ARROW_DOWN
arrow_array[0][1][0][0] = ARROW_UP
arrow_array[0][0][1][0] = ARROW_RIGHT
arrow_array[0][0][0][1] = ARROW_LEFT
arrow_array[1][0][1][0] = ARROW_SOUTHEAST
arrow_array[1][0][0][1] = ARROW_SOUTHWEST
arrow_array[0][1][1][0] = ARROW_NORTHEAST
arrow_array[0][1][0][1] = ARROW_NORTHWEST


class PolicyIteration:
    def __init__(self, env: Env):
        self.env = env
        # state에 대한 table 생성
        self.value_table = np.zeros(env.observation_space.n)
        # state, action에 대한 table 생성
        self.policy_table = np.ones([env.observation_space.n, env.action_space.n]) / env.action_space.n

    def policy_evaluation(self, iter_num, discount_factor=0.99):
        # 주어진 횟수만큼 policy evaluation 수행
        for _ in range(iter_num):
            next_value_table = np.zeros_like(self.value_table)

            for state in range(self.env.observation_space.n):
                # 종료 상태는 건너뜀
                if self.is_terminated(state):
                    continue

                value = 0.0
                policy = self.get_policy(state)
                # 각 action에 대해 기대값 계산
                for action in range(self.env.action_space.n):
                    _, next_state, reward, _ = self.env.P[state][action][0]
                    
                    value += policy[action] * (reward + discount_factor * self.get_value(next_state))
                next_value_table[state] = value

            self.value_table = next_value_table

    def policy_improvement(self, discount_factor=0.99):
        # 모든 state에 대해 policy를 개선
        for state in range(self.env.observation_space.n):
            # 종료 상태는 건너뜀
            if self.is_terminated(state):
                continue

            # 각 action에 대한 가치 계산
            action_values = np.zeros(self.env.action_space.n)
            for action in range(self.env.action_space.n):
                # 환경의 transition 정보로 다음 상태와 보상 확인
                _, next_state, reward, _ = self.env.P[state][action][0]
                action_values[action] = reward + discount_factor * self.get_value(next_state)

            # 최대 가치의 action들에 대해 확률 동일하게 할당 (greedy)
            max_value = np.max(action_values)
            max_actions = np.flatnonzero(action_values == max_value)
            policy = np.zeros(self.env.action_space.n)
            policy[max_actions] = 1.0 / len(max_actions)
            self.policy_table[state] = policy

    def Q4_policy_evaluation(self, delta=0.00001, discount_factor=0.99):
        value_stable = True
        # max 변화량 < delta 일 때까지 policy evaluation 수행
        total_episode = 0

        while True:
            total_episode += 1
            next_value_table = np.zeros_like(self.value_table)

            for state in range(self.env.observation_space.n):
                # 종료 상태는 건너뜀
                if self.is_terminated(state):
                    continue

                value = 0.0
                policy = self.get_policy(state)
                # 각 action에 대해 기대값 계산
                for action in range(self.env.action_space.n):
                    _, next_state, reward, _ = self.env.P[state][action][0]

                    current_row, current_col, current_passenger_position, _ = self.env.decode(state)
                    next_row, next_col, next_passenger_position, _ = self.env.decode(next_state)
                    if current_passenger_position != 4 and next_passenger_position == 4:
                        print("pickup")
                        reward = 10

                    value += policy[action] * (reward + discount_factor * self.get_value(next_state))
                next_value_table[state] = value

            # np 배열 연산으로 변화량 계산
            max_diff = np.max(np.abs(next_value_table - self.value_table))
            self.value_table = next_value_table
            if max_diff < delta:
                print(f'total episodes: {total_episode}')
                break

    def Q4_policy_improvement(self, discount_factor=0.99):
        policy_stable = True
        # 모든 state에 대해 policy를 개선
        for state in range(self.env.observation_space.n):
            # 종료 상태는 건너뜀
            if self.is_terminated(state):
                continue

            # 각 action에 대한 가치 계산
            action_values = np.zeros(self.env.action_space.n)
            for action in range(self.env.action_space.n):
                # 환경의 transition 정보로 다음 상태와 보상 확인
                _, next_state, reward, _ = self.env.P[state][action][0]
                action_values[action] = reward + discount_factor * self.get_value(next_state)

            # 최대 가치의 action들에 대해 확률 동일하게 할당 (greedy)
            max_value = np.max(action_values)
            max_actions = np.flatnonzero(action_values == max_value)
            policy = np.zeros(self.env.action_space.n)
            policy[max_actions] = 1.0 / len(max_actions)
            if not np.array_equal(policy, self.policy_table[state]):
                policy_stable = False
            self.policy_table[state] = policy

        return policy_stable
        
    def print_value(self, state):
        if self.is_terminated(state):
            print("This state is terminated")
            return
        print(f'value: {self.get_value(state)}')

    def print_board(self, state, action_mask):
        print('Before pickup: ')
        board = [   '+---------+',
                    '|R: | : :G|',
                    '| : | : : |',
                    '| : : : : |',
                    '| | : | : |',
                    '|Y| : |B: |',
                    '+---------+']
        _, _, passenger_location, destination = self.env.decode(state)
        for row, line in enumerate(board):
            for col, cell in enumerate(line):
                if cell == ' ':
                    temp_state = self.env.encode(row - 1, col // 2, passenger_location, destination)
                    policy = self.get_policy(temp_state)
                    policy_mask = []
                    for index in range(4):
                        if policy[index] != 0 and action_mask[index] != 0:
                            policy_mask.append(1)
                        else:
                            policy_mask.append(0)
                    
                    print(arrow_array[policy_mask[0]][policy_mask[1]][policy_mask[2]][policy_mask[3]], end = '')
                else:
                    print(cell, end = '')
            print()
        print()
        print()

        print('after pickup: ')
        for row, line in enumerate(board):
            for col, cell in enumerate(line):
                if cell == ' ':
                    temp_state = self.env.encode(row - 1, col // 2, 4, destination)
                    policy = self.get_policy(temp_state)
                    policy_mask = []
                    for index in range(4):
                        if policy[index] != 0 and action_mask[index] != 0:
                            policy_mask.append(1)
                        else:
                            policy_mask.append(0)
                    
                    print(arrow_array[policy_mask[0]][policy_mask[1]][policy_mask[2]][policy_mask[3]], end = '')
                else:
                    print(cell, end = '')
            print()
        print()

    def get_policy(self, state, debug = False):
        policy = self.policy_table[state]
        if debug:
            self.print_policy(policy)
        return policy
    
    def print_policy(self, policy):
        print('policy: {', end = '')
        for action in range(len(action_names)):
            if policy[action] != 0:
                print(f'{action_names[action]}: {policy[action]}', end = ' ')
        print('}')

    def get_value(self, state):
        return self.value_table[state]
    
    def get_action(self, state, debug = False):
        policy = self.get_policy(state)
        action = np.random.choice(len(policy), p=policy)
        if debug:
            self.print_action(action)
        return action
    
    def print_action(self, action):
        print(f'action: {action_names[action]}')
    
    def is_terminated(self, state):
        _, _, passenger_location, destination = self.env.decode(state)

        if passenger_location == destination:
            return True
        
        return False

