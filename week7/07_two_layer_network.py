"""
==========================================
07. 2층 신경망 (Simple 2-Layer Neural Network)
==========================================
강의 페이지 19 내용

단 20줄로 만드는 2층 신경망!
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randn


def simple_2layer_network():
    """
    강의 페이지 19의 20줄 신경망 코드
    """
    print("=" * 60)
    print("페이지 19: 20줄 신경망 코드 실행")
    print("=" * 60)
    
    # 1. 신경망 정의
    N, D_in, H, D_out = 64, 1000, 100, 10
    print(f"\n네트워크 구조:")
    print(f"  - 배치 크기 N: {N}")
    print(f"  - 입력 차원: {D_in}")
    print(f"  - 은닉 노드 수: {H}")
    print(f"  - 출력 차원: {D_out}")
    
    np.random.seed(42)
    x, y = randn(N, D_in), randn(N, D_out)
    w1, w2 = randn(D_in, H), randn(H, D_out)
    
    losses = []
    
    print(f"\n학습 시작...")
    
    # 2. 학습 루프
    for t in range(2000):
        # 순방향
        h = 1 / (1 + np.exp(-x.dot(w1)))   # 시그모이드
        y_pred = h.dot(w2)
        loss = np.square(y_pred - y).sum()
        losses.append(loss)
        
        if t % 200 == 0:
            print(f"  Epoch {t:>4}: Loss = {loss:.4f}")
        
        # 역전파 (기울기 계산)
        grad_y_pred = 2.0 * (y_pred - y)
        grad_w2 = h.T.dot(grad_y_pred)
        grad_h = grad_y_pred.dot(w2.T)
        grad_w1 = x.T.dot(grad_h * h * (1 - h))
        
        # 경사하강법
        w1 -= 1e-4 * grad_w1
        w2 -= 1e-4 * grad_w2
    
    print(f"\n최종 Loss: {losses[-1]:.4f}")
    
    # 학습 곡선 그리기
    plt.figure(figsize=(10, 5))
    plt.plot(losses, 'b-', linewidth=1.5)
    plt.title('20-Line Neural Network Training', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.savefig('/home/claude/neural_network_examples/simple_2layer.png', dpi=80)
    plt.close()
    print("그래프 저장됨: simple_2layer.png")


class TwoLayerNet:
    """
    2층 신경망 클래스 (객체 지향 버전)
    """
    
    def __init__(self, input_size, hidden_size, output_size, weight_init='random'):
        """
        신경망 초기화
        
        Args:
            input_size: 입력 차원
            hidden_size: 은닉 노드 수
            output_size: 출력 차원
            weight_init: 가중치 초기화 방법 ('random', 'xavier', 'he')
        """
        self.params = {}
        
        if weight_init == 'random':
            # 일반 정규분포
            self.params['W1'] = 0.01 * np.random.randn(input_size, hidden_size)
            self.params['W2'] = 0.01 * np.random.randn(hidden_size, output_size)
        elif weight_init == 'xavier':
            # Xavier 초기화 (sigmoid, tanh용)
            self.params['W1'] = np.random.randn(input_size, hidden_size) / np.sqrt(input_size)
            self.params['W2'] = np.random.randn(hidden_size, output_size) / np.sqrt(hidden_size)
        elif weight_init == 'he':
            # He 초기화 (ReLU용)
            self.params['W1'] = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
            self.params['W2'] = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        
        self.params['b1'] = np.zeros(hidden_size)
        self.params['b2'] = np.zeros(output_size)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def predict(self, x):
        """순방향 계산"""
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        
        a1 = np.dot(x, W1) + b1
        z1 = self.sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = self.softmax(a2)
        
        return y
    
    def loss(self, x, t):
        """손실 함수 (교차 엔트로피)"""
        y = self.predict(x)
        delta = 1e-7
        return -np.sum(t * np.log(y + delta)) / x.shape[0]
    
    def accuracy(self, x, t):
        """정확도"""
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        return np.sum(y == t) / float(x.shape[0])
    
    def gradient(self, x, t):
        """역전파로 기울기 계산"""
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        
        grads = {}
        batch_num = x.shape[0]
        
        # 순방향
        a1 = np.dot(x, W1) + b1
        z1 = self.sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = self.softmax(a2)
        
        # 역방향
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)
        
        da1 = np.dot(dy, W2.T)
        dz1 = z1 * (1 - z1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)
        
        return grads


def train_xor():
    """XOR 문제를 2층 신경망으로 학습"""
    print("\n" + "=" * 60)
    print("XOR 문제 학습 (2층 신경망)")
    print("=" * 60)
    
    # XOR 데이터
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    
    # one-hot encoding
    t = np.array([
        [1, 0],  # 0
        [0, 1],  # 1
        [0, 1],  # 1
        [1, 0]   # 0
    ])
    
    # 신경망 생성
    np.random.seed(42)
    net = TwoLayerNet(input_size=2, hidden_size=4, output_size=2, weight_init='xavier')
    
    # 학습
    learning_rate = 0.5
    losses = []
    
    print(f"\n{'Epoch':>5} {'Loss':>10} {'Accuracy':>10}")
    print("-" * 30)
    
    for epoch in range(10001):
        grads = net.gradient(X, t)
        
        for key in net.params.keys():
            net.params[key] -= learning_rate * grads[key]
        
        loss = net.loss(X, t)
        losses.append(loss)
        
        if epoch % 1000 == 0:
            acc = net.accuracy(X, t)
            print(f"{epoch:>5} {loss:>10.4f} {acc:>10.2%}")
    
    # 최종 결과
    print(f"\n최종 예측:")
    print(f"{'x1':>3} {'x2':>3} | {'정답':>6} {'예측':>6}")
    print("-" * 30)
    
    y_pred = net.predict(X)
    for i in range(len(X)):
        true_label = np.argmax(t[i])
        pred_label = np.argmax(y_pred[i])
        print(f"{X[i][0]:>3} {X[i][1]:>3} |  {true_label:>5} {pred_label:>6}")
    
    # 학습 곡선
    plt.figure(figsize=(10, 5))
    plt.plot(losses, 'b-', linewidth=1)
    plt.title('XOR Problem Learning (2-Layer Neural Network)', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig('/home/claude/neural_network_examples/xor_learning.png', dpi=80)
    plt.close()
    print("\n그래프 저장됨: xor_learning.png")
    
    return net


def visualize_xor_decision(net):
    """XOR의 결정 경계 시각화"""
    x1_range = np.linspace(-0.5, 1.5, 100)
    x2_range = np.linspace(-0.5, 1.5, 100)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    
    # 격자점 예측
    grid_points = np.c_[X1.ravel(), X2.ravel()]
    predictions = net.predict(grid_points)
    Z = np.argmax(predictions, axis=1).reshape(X1.shape)
    
    plt.figure(figsize=(8, 8))
    plt.contourf(X1, X2, Z, levels=[-0.5, 0.5, 1.5],
                 colors=['lightblue', 'lightyellow'], alpha=0.5)
    
    # 학습 데이터 표시
    plt.scatter(0, 0, c='blue', s=300, marker='x', linewidth=4, label='y=0')
    plt.scatter(1, 1, c='blue', s=300, marker='x', linewidth=4)
    plt.scatter(0, 1, c='red', s=300, marker='o', edgecolors='black', linewidth=2, label='y=1')
    plt.scatter(1, 0, c='red', s=300, marker='o', edgecolors='black', linewidth=2)
    
    plt.xlabel('x1', fontsize=12)
    plt.ylabel('x2', fontsize=12)
    plt.title('XOR Decision Boundary (Learned)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(-0.5, 1.5)
    plt.ylim(-0.5, 1.5)
    plt.savefig('/home/claude/neural_network_examples/xor_boundary.png', dpi=80)
    plt.close()
    print("그래프 저장됨: xor_boundary.png")


if __name__ == "__main__":
    # 강의 예제: 20줄 신경망
    simple_2layer_network()
    
    # XOR 학습
    net = train_xor()
    visualize_xor_decision(net)
    
    print("\n" + "=" * 60)
    print("핵심 정리:")
    print("- 단 20줄로도 신경망 구현 가능!")
    print("- XOR 같은 비선형 문제도 2층이면 풀 수 있어요")
    print("- 객체지향 코드로 더 깔끔하게 만들 수 있어요")
    print("=" * 60)
