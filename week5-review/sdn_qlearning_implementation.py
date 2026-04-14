"""
Q-Learning 기반 SDN 라우팅 최적화 구현
A Q-Learning based Routing Optimization Model in a Software Defined Network

주요 기능:
- Q-Learning을 이용한 최적 라우팅 경로 학습
- 대역폭과 지연시간을 고려한 보상 함수
- 네트워크 시뮬레이션 및 성능 평가
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random
from typing import Dict, List, Tuple

class NetworkEnvironment:
    """
    네트워크 환경 정의
    - 노드(라우터)와 링크로 구성된 그래프
    - 각 링크는 대역폭과 지연시간 정보 보유
    """
    
    def __init__(self, num_nodes: int = 6):
        self.num_nodes = num_nodes
        self.current_node = 0
        self.destination = num_nodes - 1
        
        # 네트워크 토폴로지: (출발지, 도착지) -> (대역폭, 지연시간)
        self.links = {
            (0, 1): {'bandwidth': 10, 'delay': 5},
            (0, 2): {'bandwidth': 8, 'delay': 8},
            (1, 3): {'bandwidth': 6, 'delay': 10},
            (1, 4): {'bandwidth': 7, 'delay': 7},
            (2, 3): {'bandwidth': 9, 'delay': 6},
            (2, 4): {'bandwidth': 5, 'delay': 12},
            (3, 5): {'bandwidth': 8, 'delay': 4},
            (4, 5): {'bandwidth': 10, 'delay': 3},
            (1, 2): {'bandwidth': 12, 'delay': 2},
        }
        
        # 인접 리스트 생성
        self.adjacency = defaultdict(list)
        for (src, dst) in self.links:
            self.adjacency[src].append(dst)
    
    def reset(self, source: int = 0, dest: int = None) -> int:
        """에피소드 리셋"""
        self.current_node = source
        if dest is None:
            self.destination = self.num_nodes - 1
        else:
            self.destination = dest
        return self.current_node
    
    def get_available_actions(self, node: int) -> List[int]:
        """현재 노드에서 가능한 다음 노드"""
        return self.adjacency[node]
    
    def step(self, action: int) -> Tuple[int, float, bool]:
        """
        행동 실행
        
        Returns:
            next_state: 다음 노드
            reward: 보상값
            done: 종료 여부
        """
        if action not in self.get_available_actions(self.current_node):
            return self.current_node, -10, False
        
        # 보상 계산
        reward = self._calculate_reward(self.current_node, action)
        
        # 다음 상태 업데이트
        prev_node = self.current_node
        self.current_node = action
        
        # 종료 조건 (목적지 도달)
        done = (self.current_node == self.destination)
        
        # 목적지 도달 보너스
        if done:
            reward += 10
        
        return self.current_node, reward, done
    
    def _calculate_reward(self, src: int, dst: int) -> float:
        """
        보상 함수: 대역폭과 지연시간 고려
        Reward = w1 * (BW/BW_max) + w2 * (1 - delay/delay_max)
        """
        if (src, dst) not in self.links:
            return -10
        
        link_info = self.links[(src, dst)]
        bandwidth = link_info['bandwidth']
        delay = link_info['delay']
        
        # 가중치
        w1 = 0.5  # 대역폭 가중치
        w2 = 0.5  # 지연 가중치
        
        # 정규화
        max_bandwidth = 12
        max_delay = 12
        
        bandwidth_score = bandwidth / max_bandwidth
        delay_score = 1 - (delay / max_delay)
        
        reward = w1 * bandwidth_score + w2 * delay_score
        
        return reward


class QLearningAgent:
    """
    Q-Learning 에이전트 구현
    - Q-table: (상태, 행동) -> Q값 저장
    - ε-Greedy 전략으로 탐험/활용 균형
    """
    
    def __init__(self, num_nodes: int, alpha: float = 0.1, gamma: float = 0.9, 
                 epsilon: float = 0.1):
        """
        Args:
            num_nodes: 네트워크 노드 개수
            alpha: 학습률 (0~1)
            gamma: 할인율 (0~1)
            epsilon: 탐험률 (0~1)
        """
        self.num_nodes = num_nodes
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Q-table 초기화: {(state, action): Q값}
        self.q_table = defaultdict(lambda: defaultdict(float))
        
        # 에피소드별 누적 보상 기록
        self.episode_rewards = []
        
    def select_action(self, state: int, available_actions: List[int]) -> int:
        """
        ε-Greedy 전략으로 행동 선택
        
        확률 ε로 탐험 (무작위), 확률 (1-ε)로 활용 (최적)
        """
        if not available_actions:
            return state
        
        if random.random() < self.epsilon:
            # 탐험: 무작위 선택
            return random.choice(available_actions)
        else:
            # 활용: 최적 행동 선택
            q_values = [self.q_table[state][action] for action in available_actions]
            max_q = max(q_values)
            
            # 여러 최적값이 있으면 무작위 선택
            best_actions = [available_actions[i] for i, q in enumerate(q_values) 
                          if q == max_q]
            return random.choice(best_actions)
    
    def update_q_value(self, state: int, action: int, reward: float, 
                      next_state: int, next_available_actions: List[int]):
        """
        Q-value 업데이트 (Policy Control)
        
        Q(s,a) ← Q(s,a) + α[r + γ·max_a' Q(s',a') - Q(s,a)]
        """
        # 다음 상태의 최대 Q값 계산
        if next_available_actions:
            max_next_q = max([self.q_table[next_state][a] 
                             for a in next_available_actions])
        else:
            max_next_q = 0
        
        # Q값 업데이트
        current_q = self.q_table[state][action]
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        
        self.q_table[state][action] = new_q
    
    def train(self, env: NetworkEnvironment, num_episodes: int = 500) -> List[float]:
        """
        에이전트 학습
        
        Args:
            env: 네트워크 환경
            num_episodes: 학습 에피소드 수
            
        Returns:
            에피소드별 누적 보상
        """
        episode_rewards = []
        
        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            max_steps = 20
            step_count = 0
            
            while step_count < max_steps:
                # 행동 선택
                available_actions = env.get_available_actions(state)
                if not available_actions:
                    break
                
                action = self.select_action(state, available_actions)
                
                # 환경에서 실행
                next_state, reward, done = env.step(action)
                episode_reward += reward
                
                # Q-value 업데이트
                next_available_actions = env.get_available_actions(next_state)
                self.update_q_value(state, action, reward, next_state, 
                                  next_available_actions)
                
                state = next_state
                step_count += 1
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            
            # 학습률 감소
            if episode > 0 and episode % 50 == 0:
                self.epsilon *= 0.95
        
        self.episode_rewards = episode_rewards
        return episode_rewards
    
    def get_optimal_path(self, env: NetworkEnvironment, source: int = 0, 
                        dest: int = None) -> Tuple[List[int], float]:
        """
        학습된 Q-table을 이용해 최적 경로 생성
        
        Returns:
            경로 (노드 리스트), 누적 보상
        """
        if dest is None:
            dest = env.num_nodes - 1
        
        path = [source]
        current_state = source
        total_reward = 0
        max_steps = 20
        step_count = 0
        
        while current_state != dest and step_count < max_steps:
            available_actions = env.get_available_actions(current_state)
            
            if not available_actions:
                break
            
            # 최적 행동 선택 (활용만)
            q_values = [self.q_table[current_state][action] 
                       for action in available_actions]
            best_action = available_actions[np.argmax(q_values)]
            
            # 이동
            next_state = best_action
            path.append(next_state)
            
            # 보상 계산
            if (current_state, next_state) in env.links:
                link_info = env.links[(current_state, next_state)]
                reward = env._calculate_reward(current_state, next_state)
                total_reward += reward
            
            current_state = next_state
            step_count += 1
        
        return path, total_reward


class BaselineRouter:
    """
    기준선 라우터 (Shortest Path)
    비교를 위한 기존 알고리즘
    """
    
    def __init__(self, env: NetworkEnvironment):
        self.env = env
    
    def find_shortest_path(self, source: int, dest: int) -> Tuple[List[int], float]:
        """BFS를 이용한 최단 경로 찾기"""
        from collections import deque
        
        queue = deque([(source, [source], 0)])
        visited = {source}
        best_path = None
        best_reward = -float('inf')
        
        while queue:
            current, path, reward = queue.popleft()
            
            if current == dest:
                if reward > best_reward:
                    best_reward = reward
                    best_path = path
                continue
            
            for next_node in self.env.get_available_actions(current):
                if next_node not in visited or next_node == dest:
                    next_reward = self.env._calculate_reward(current, next_node)
                    queue.append((next_node, path + [next_node], reward + next_reward))
                    visited.add(next_node)
        
        return best_path if best_path else [source, dest], best_reward


def run_simulation():
    """전체 시뮬레이션 실행"""
    
    print("=" * 70)
    print("Q-Learning 기반 SDN 라우팅 최적화 시뮬레이션")
    print("=" * 70)
    
    # 환경 및 에이전트 생성
    env = NetworkEnvironment(num_nodes=6)
    agent = QLearningAgent(num_nodes=6, alpha=0.1, gamma=0.9, epsilon=0.3)
    baseline = BaselineRouter(env)
    
    # 학습
    print("\n[1단계] Q-Learning 에이전트 학습 중...")
    rewards = agent.train(env, num_episodes=500)
    print(f"✓ 500 에피소드 학습 완료")
    print(f"  - 초기 평균 보상: {np.mean(rewards[:50]):.4f}")
    print(f"  - 최종 평균 보상: {np.mean(rewards[-50:]):.4f}")
    
    # 최적 경로 찾기
    print("\n[2단계] 최적 경로 계산 중...")
    ql_path, ql_reward = agent.get_optimal_path(env, source=0, dest=5)
    base_path, base_reward = baseline.find_shortest_path(0, 5)
    
    print("\n📊 결과 비교:")
    print("-" * 70)
    print(f"{'방법':<20} {'경로':<30} {'누적 보상':<15}")
    print("-" * 70)
    print(f"{'Q-Learning':<20} {str(ql_path):<30} {ql_reward:.4f}")
    print(f"{'Shortest Path':<20} {str(base_path):<30} {base_reward:.4f}")
    print("-" * 70)
    
    improvement = ((ql_reward - base_reward) / abs(base_reward) * 100) if base_reward != 0 else 0
    print(f"\n✓ Q-Learning 성능 개선율: {improvement:.2f}%")
    
    # 상세 경로 정보
    print("\n[3단계] 경로 상세 정보:")
    print("\n▶ Q-Learning 경로:")
    print_path_details(env, ql_path)
    
    print("\n▶ Shortest Path 경로:")
    print_path_details(env, base_path)
    
    # Q-table 일부 출력
    print("\n[4단계] 학습된 Q-table 샘플 (상태별 최대 Q값):")
    print("-" * 70)
    for state in range(6):
        available_actions = env.get_available_actions(state)
        if available_actions:
            max_q_value = max([agent.q_table[state][a] for a in available_actions])
            best_action = max(available_actions, 
                            key=lambda a: agent.q_table[state][a])
            print(f"상태 {state}: 최적 다음 노드 = {best_action}, "
                  f"Q값 = {max_q_value:.4f}")
    
    # 학습 곡선 그리기
    plot_learning_curve(rewards)
    
    print("\n" + "=" * 70)
    print("시뮬레이션 완료!")
    print("=" * 70)


def print_path_details(env: NetworkEnvironment, path: List[int]):
    """경로 상세 정보 출력"""
    total_bandwidth = 0
    total_delay = 0
    total_reward = 0
    
    print(f"경로: {' → '.join(map(str, path))}")
    print("-" * 70)
    print(f"{'링크':<10} {'대역폭':<12} {'지연(ms)':<12} {'보상':<10}")
    print("-" * 70)
    
    for i in range(len(path) - 1):
        src, dst = path[i], path[i + 1]
        if (src, dst) in env.links:
            link_info = env.links[(src, dst)]
            bw = link_info['bandwidth']
            delay = link_info['delay']
            reward = env._calculate_reward(src, dst)
            
            print(f"{src}→{dst:<8} {bw:<12} {delay:<12} {reward:.4f}")
            
            total_bandwidth += bw
            total_delay += delay
            total_reward += reward
    
    print("-" * 70)
    print(f"{"합계":<10} {total_bandwidth:<12} {total_delay:<12} {total_reward:.4f}")


def plot_learning_curve(rewards: List[float]):
    """학습 곡선 시각화"""
    
    # 이동 평균 계산
    window_size = 50
    moving_avg = [np.mean(rewards[max(0, i-window_size):i+1]) 
                  for i in range(len(rewards))]
    
    plt.figure(figsize=(12, 6))
    
    # 원본 보상
    plt.plot(rewards, alpha=0.3, label='Raw Rewards', color='blue')
    
    # 이동 평균
    plt.plot(moving_avg, label=f'Moving Average (window={window_size})', 
             color='red', linewidth=2)
    
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Cumulative Reward', fontsize=12)
    plt.title('Q-Learning Agent Learning Curve', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 이미지 저장
    plt.savefig('/home/claude/learning_curve.png', dpi=100, bbox_inches='tight')
    print("\n📈 학습 곡선이 'learning_curve.png'로 저장되었습니다.")
    plt.close()


if __name__ == "__main__":
    run_simulation()
