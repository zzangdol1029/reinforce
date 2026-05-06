"""
==========================================
06. 역전파 (Back Propagation) 예제
==========================================
강의 페이지 17-18 내용

3계층 신경망의 역전파 학습 과정을 단계별로 구현합니다.
"""

import numpy as np


def sigmoid(x):
    """시그모이드 함수"""
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(y):
    """시그모이드의 도함수: y * (1 - y)"""
    return y * (1 - y)


def forward_pass(X, V, W):
    """
    순방향 계산
    
    Args:
        X: 입력 (1차원 배열)
        V: 입력→은닉 가중치
        W: 은닉→출력 가중치
    
    Returns:
        NET_z, Z, NET_y, y
    """
    # 입력 → 은닉층
    NET_z = np.dot(X, V.T)
    Z = sigmoid(NET_z)
    
    # 은닉 → 출력층
    NET_y = np.dot(Z, W.T)
    y = sigmoid(NET_y)
    
    return NET_z, Z, NET_y, y


def backward_pass(X, Z, y, t, W, V, learning_rate=1.0):
    """
    역방향 계산 (가중치 업데이트)
    
    Args:
        X: 입력
        Z: 은닉층 출력
        y: 출력층 출력
        t: 목표값
        W, V: 가중치
        learning_rate: 학습률
    
    Returns:
        업데이트된 V, W
    """
    # 출력층 에러 신호
    delta_y = (t - y) * sigmoid_derivative(y)
    
    # W 업데이트 (은닉 → 출력)
    delta_W = learning_rate * delta_y * Z
    
    # 은닉층 에러 신호
    delta_z = np.zeros(Z.shape)
    for i in range(len(Z)):
        delta_z[i] = Z[i] * (1 - Z[i]) * delta_y * W[i]
    
    # V 업데이트 (입력 → 은닉)
    delta_V = learning_rate * np.outer(delta_z, X)
    
    return delta_V, delta_W, delta_y, delta_z


def lecture_example():
    """강의 페이지 17-18 예제"""
    print("=" * 60)
    print("강의 예제: 3계층 신경망 역전파")
    print("=" * 60)
    
    # 초기 가중치
    V = np.array([
        [0.1, 0.2, -0.5],   # 첫 번째 은닉 노드
        [0.3, 0.1, -0.3]    # 두 번째 은닉 노드
    ])
    
    W = np.array([0.1, 0.2])  # 출력층 가중치
    
    # 학습 패턴 A
    X = np.array([1, 0, 0])
    t = -1  # 목표값
    learning_rate = 1.0
    
    print(f"\n초기 설정:")
    print(f"V = \n{V}")
    print(f"W = {W}")
    print(f"X = {X}")
    print(f"t = {t}")
    print(f"학습률 α = {learning_rate}")
    
    # ============================================
    # Step 1: 순방향 계산
    # ============================================
    print("\n" + "─" * 60)
    print("Step 1: 순방향 계산 (Forward Pass)")
    print("─" * 60)
    
    NET_z, Z, NET_y, y = forward_pass(X, V, W)
    
    print(f"\n1-1. 은닉층 입력 NET_z = X · V^T")
    print(f"     NET_z = {NET_z.round(4)}")
    
    print(f"\n1-2. 은닉층 출력 Z = sigmoid(NET_z)")
    print(f"     Z = {Z.round(4)}")
    
    print(f"\n1-3. 출력층 입력 NET_y = Z · W^T")
    print(f"     NET_y = {NET_y:.4f}")
    
    print(f"\n1-4. 출력층 출력 y = sigmoid(NET_y)")
    print(f"     y = {y:.4f}")
    
    # ============================================
    # Step 2: 오차 계산
    # ============================================
    print("\n" + "─" * 60)
    print("Step 2: 오차 계산")
    print("─" * 60)
    
    E = 0.5 * (t - y) ** 2
    print(f"\nE = ½(t - y)² = 0.5 × ({t} - {y:.4f})² = {E:.4f}")
    
    # ============================================
    # Step 3: 역방향 계산 - 출력층
    # ============================================
    print("\n" + "─" * 60)
    print("Step 3: 출력층 가중치 업데이트")
    print("─" * 60)
    
    delta_y = (t - y) * y * (1 - y)
    print(f"\n3-1. 출력층 에러 신호:")
    print(f"     δy = (t - y) × y × (1 - y)")
    print(f"     δy = ({t} - {y:.4f}) × {y:.4f} × (1 - {y:.4f})")
    print(f"     δy = {delta_y:.4f}")
    
    delta_W = learning_rate * delta_y * Z
    print(f"\n3-2. 가중치 변화량:")
    print(f"     ΔW = α × δy × Z")
    print(f"     ΔW = {learning_rate} × {delta_y:.4f} × {Z.round(4)}")
    print(f"     ΔW = {delta_W.round(4)}")
    
    W_new = W + delta_W
    print(f"\n3-3. 업데이트된 W:")
    print(f"     W_new = W + ΔW = {W_new.round(4)}")
    
    # ============================================
    # Step 4: 역방향 계산 - 은닉층
    # ============================================
    print("\n" + "─" * 60)
    print("Step 4: 은닉층 가중치 업데이트")
    print("─" * 60)
    
    delta_z = np.zeros(Z.shape)
    for i in range(len(Z)):
        delta_z[i] = Z[i] * (1 - y) * delta_y * W[i]
    
    print(f"\n4-1. 은닉층 에러 신호:")
    for i in range(len(Z)):
        print(f"     δz_{i+1} = z_{i+1} × (1-y) × δy × w_{i+1}")
        print(f"            = {Z[i]:.4f} × (1-{y:.4f}) × {delta_y:.4f} × {W[i]}")
        print(f"            = {delta_z[i]:.4f}")
    
    delta_V = learning_rate * np.outer(delta_z, X)
    print(f"\n4-2. 가중치 변화량:")
    print(f"     ΔV = α × δz × X")
    print(f"     ΔV = \n{delta_V.round(4)}")
    
    V_new = V + delta_V
    print(f"\n4-3. 업데이트된 V:")
    print(f"     V_new = \n{V_new.round(4)}")
    
    return V_new, W_new


def train_iteratively():
    """반복 학습 예제"""
    print("\n" + "=" * 60)
    print("반복 학습 예제 (100 epochs)")
    print("=" * 60)
    
    # 초기 가중치
    V = np.array([
        [0.1, 0.2, -0.5],
        [0.3, 0.1, -0.3]
    ])
    W = np.array([0.1, 0.2])
    
    # 입력과 목표값
    X = np.array([1, 0, 0])
    t = -1
    learning_rate = 0.5
    
    errors = []
    
    print(f"\n{'Epoch':>5} {'y':>10} {'Error':>10}")
    print("-" * 30)
    
    for epoch in range(101):
        # 순방향
        _, Z, _, y = forward_pass(X, V, W)
        
        # 오차
        error = 0.5 * (t - y) ** 2
        errors.append(error)
        
        if epoch % 10 == 0:
            print(f"{epoch:>5} {y:>10.4f} {error:>10.6f}")
        
        # 역방향
        delta_V, delta_W, _, _ = backward_pass(X, Z, y, t, W, V, learning_rate)
        V = V + delta_V
        W = W + delta_W
    
    return V, W, errors


def visualize_learning_curve():
    """학습 곡선 시각화"""
    import matplotlib.pyplot as plt
    
    V, W, errors = train_iteratively()
    
    plt.figure(figsize=(10, 5))
    plt.plot(errors, 'b-', linewidth=2)
    plt.title('Learning Curve (Backpropagation)', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Error', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig('/home/claude/neural_network_examples/backpropagation_curve.png', dpi=80)
    plt.close()
    print("\n그래프 저장됨: backpropagation_curve.png")


if __name__ == "__main__":
    # 강의 예제 단계별 풀이
    V_new, W_new = lecture_example()
    
    # 반복 학습 시각화
    visualize_learning_curve()
    
    print("\n" + "=" * 60)
    print("핵심 정리:")
    print("- 순방향: 입력 → 은닉 → 출력 (예측 계산)")
    print("- 역방향: 출력 → 은닉 → 입력 (에러 전파)")
    print("- 각 층의 가중치를 업데이트하면서 에러 감소")
    print("- 반복할수록 정답에 가까워짐!")
    print("=" * 60)
