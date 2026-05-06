"""
==========================================
05. 퍼셉트론 학습 예제
==========================================
강의 페이지 13-14 내용

퍼셉트론의 학습 규칙:
ΔW = η × (정답 - 출력) × 입력
W_new = W_old + ΔW
"""

import numpy as np
import matplotlib.pyplot as plt


def perceptron_learning_step(W, X, t, y, learning_rate=1.0):
    """
    퍼셉트론 학습 한 단계
    
    Args:
        W: 현재 가중치
        X: 입력
        t: 목표값 (정답)
        y: 현재 출력
        learning_rate: 학습률
    
    Returns:
        ΔW (가중치 변화량), W_new (업데이트된 가중치)
    """
    delta_W = learning_rate * (t - y) * X
    W_new = W + delta_W
    return delta_W, W_new


def example_lecture():
    """강의 페이지 14의 예제 풀이"""
    print("=" * 50)
    print("강의 예제 (페이지 14)")
    print("=" * 50)
    print("\n조건:")
    print("- 현재 가중치 W = [0.2, 0.5, 0.3]")
    print("- 입력 X = [0, 0, 1]")
    print("- 목표값 t = 1")
    print("- 출력 y = -1")
    print("- 학습률 η = 1")
    
    # 주어진 조건
    W = np.array([0.2, 0.5, 0.3])
    X = np.array([0, 0, 1])
    t = 1
    y = -1
    eta = 1.0
    
    print(f"\n--- 풀이 ---")
    
    # ΔW 계산
    delta_W = eta * (t - y) * X
    print(f"\n1. 가중치 변화량 ΔW 계산:")
    print(f"   ΔW = η × (t - y) × X")
    print(f"   ΔW = {eta} × ({t} - ({y})) × {X}")
    print(f"   ΔW = {eta} × {t - y} × {X}")
    print(f"   ΔW = {delta_W}")
    
    # 새 가중치 계산
    W_new = W + delta_W
    print(f"\n2. 새 가중치 W_new 계산:")
    print(f"   W_new = W + ΔW")
    print(f"   W_new = {W} + {delta_W}")
    print(f"   W_new = {W_new}")
    
    return W_new


def perceptron_train_AND():
    """AND 게이트 학습 예제"""
    print("\n" + "=" * 50)
    print("AND 게이트 학습")
    print("=" * 50)
    
    # 학습 데이터 (편향 포함)
    # X = [bias, x1, x2]
    X_train = np.array([
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1]
    ])
    
    # 정답 (AND 게이트)
    t_train = np.array([0, 0, 0, 1])
    
    # 가중치 초기화
    np.random.seed(42)
    W = np.random.randn(3) * 0.1
    print(f"초기 가중치: {W}")
    
    # 학습률
    learning_rate = 0.1
    
    # 학습 과정
    epoch_errors = []
    
    for epoch in range(100):
        total_error = 0
        for i in range(len(X_train)):
            X = X_train[i]
            t = t_train[i]
            
            # 예측 (계단 함수)
            y_raw = np.dot(W, X)
            y = 1 if y_raw > 0 else 0
            
            # 가중치 업데이트
            error = t - y
            W = W + learning_rate * error * X
            total_error += abs(error)
        
        epoch_errors.append(total_error)
        
        if epoch < 10 or epoch % 10 == 0:
            print(f"Epoch {epoch+1:3d}: W = {W.round(4)}, 총 에러 = {total_error}")
        
        if total_error == 0:
            print(f"\n→ Epoch {epoch+1}에서 학습 완료!")
            break
    
    # 최종 결과 검증
    print(f"\n최종 가중치: {W.round(4)}")
    print("\n--- 학습 결과 검증 ---")
    print(f"{'x1':>3} {'x2':>3} | {'정답':>4} {'예측':>4}")
    print("-" * 25)
    for i in range(len(X_train)):
        X = X_train[i]
        t = t_train[i]
        y_raw = np.dot(W, X)
        y = 1 if y_raw > 0 else 0
        print(f"{X[1]:>3} {X[2]:>3} |  {t:>3} {y:>4}")
    
    return W, epoch_errors


def visualize_learning():
    """학습 과정 시각화"""
    W, errors = perceptron_train_AND()
    
    plt.figure(figsize=(10, 5))
    plt.plot(errors, 'b-', linewidth=2, marker='o')
    plt.title('Perceptron Learning Progress (AND Gate)', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Total Error', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig('/home/claude/neural_network_examples/perceptron_learning.png', dpi=80)
    plt.close()
    print("\n그래프 저장됨: perceptron_learning.png")


def visualize_decision_boundary():
    """학습된 퍼셉트론의 결정 경계 시각화"""
    W, _ = perceptron_train_AND()
    
    # 격자 생성
    x1_range = np.linspace(-0.5, 1.5, 100)
    x2_range = np.linspace(-0.5, 1.5, 100)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    
    # 각 점에서 예측
    Z = np.zeros_like(X1)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            x_test = np.array([1, X1[i, j], X2[i, j]])
            Z[i, j] = 1 if np.dot(W, x_test) > 0 else 0
    
    plt.figure(figsize=(8, 8))
    plt.contourf(X1, X2, Z, levels=[-0.5, 0.5, 1.5],
                 colors=['lightblue', 'lightyellow'], alpha=0.5)
    
    # 학습 데이터 점 표시
    plt.scatter(0, 0, c='blue', s=200, marker='x', linewidth=3, label='y=0')
    plt.scatter(0, 1, c='blue', s=200, marker='x', linewidth=3)
    plt.scatter(1, 0, c='blue', s=200, marker='x', linewidth=3)
    plt.scatter(1, 1, c='red', s=200, marker='o', edgecolors='black', linewidth=2, label='y=1')
    
    # 결정 경계선
    if W[2] != 0:
        x1_line = np.array([-0.5, 1.5])
        x2_line = -(W[0] + W[1] * x1_line) / W[2]
        plt.plot(x1_line, x2_line, 'g-', linewidth=2, label='Decision Boundary')
    
    plt.xlabel('x1', fontsize=12)
    plt.ylabel('x2', fontsize=12)
    plt.title(f'Learned AND Gate\nW = {W.round(3)}', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(-0.5, 1.5)
    plt.ylim(-0.5, 1.5)
    plt.savefig('/home/claude/neural_network_examples/perceptron_boundary.png', dpi=80)
    plt.close()
    print("그래프 저장됨: perceptron_boundary.png")


if __name__ == "__main__":
    # 강의 예제 실행
    example_lecture()
    
    # AND 게이트 학습
    visualize_learning()
    visualize_decision_boundary()
    
    print("\n" + "=" * 50)
    print("핵심 정리:")
    print("- 학습 규칙: ΔW = η × (정답 - 출력) × 입력")
    print("- 정답과 출력이 같으면: 가중치 변화 없음")
    print("- 정답과 출력이 다르면: 가중치 조정")
    print("- 반복하면 점점 정답에 가까워짐!")
    print("=" * 50)
