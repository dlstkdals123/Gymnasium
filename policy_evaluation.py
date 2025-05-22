import gymnasium as gym
import numpy as np
from gymnasium import Env

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

    def policy_evaluation(self, iter_num, discount_factor = 0.99):
        for episode in range(iter_num):
            next_value_table = np.zeros(self.env.observation_space.n)

            # 모든 state에 대해서 반복
            for state in range(self.env.observation_space.n):
                # 이미 끝난 state는 처리하지 않음
                if self.is_terminated(state):
                    continue

                # 벨만 기대 방정식을 통해 value 계산
                value = 0.0
                for action in range(self.env.action_space.n):
                    # self.env.P[state][action][0] = state에서 action을 했을 때 나오는 값
                    _, next_state, reward, _ = self.env.P[state][action][0]
                    next_value = self.get_value(next_state)
                    value += self.get_policy(state)[action] * (reward + discount_factor * next_value)

                next_value_table[state] = value
            
            # episode 마다 업데이트
            self.value_table = next_value_table

    def policy_improvement(self, discount_factor = 0.99):
        for state in range(self.env.observation_space.n):
            # 이미 끝난 state는 처리하지 않음
            if self.is_terminated(state):
                continue

            # max_value: 현재까지 찾은 max_value값
            # max_actions: 현재까지 찾은 max_value값에 대한 index
            max_value = float('-inf')
            max_actions = []
            policy = np.zeros(self.env.action_space.n)

            # 모든 행동에 대해서 값을 계산
            for action in range(self.env.action_space.n):
                _, next_state, reward, _ = self.env.P[state][action][0]
                value = reward + discount_factor * self.get_value(next_state)

                # 새로운 max_value가 나온 경우
                if value > max_value:
                    max_value = value
                    max_actions = [action]
                # max_value을 가진 action이 여러 개인 경우
                elif value == max_value:
                    max_actions.append(action)
            
            # max_index에 대해 uniform하게 확률 조정 후 적용
            prob = 1 / len(max_actions)
            for action in max_actions:
                policy[action] = prob

            self.policy_table[state] = policy
        
    def print_value(self, state):
        if self.is_terminated(state):
            print("This state is terminated")
            return
        print(f'value: {self.get_value(state)}')

    def print_board(self, state, action_mask):
        board = [   '+---------+',
                    '|R: | : :G|',
                    '| : | : : |',
                    '| : : : : |',
                    '| | : | : |',
                    '|Y| : |B: |',
                    '+---------+']
        _, _, passenger_location, destination = self.env.decode(state)
        if action_mask[4]:
            print('pickup...')
            print()
            return
        
        if self.is_terminated(state):
            print('This state is terminated')
            return
        
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
                    
                    # print(arrow_array[policy_mask[0]][policy_mask[1]][policy_mask[2]][policy_mask[3]], end = '')
                    print(policy, end = '')
                else:
                    print(cell, end = '')
            print()
        print()

    def get_policy(self, state):
        return self.policy_table[state]

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

