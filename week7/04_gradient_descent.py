"""
==========================================
04. 경사하강법 (Gradient Descent)
==========================================
강의 페이지 12 내용

f(x0, x1) = x0² + x1² 함수의 최솟값을 경사하강법으로 찾습니다.
산을 내려가는 것과 같은 원리입니다!
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def function_2d(x):
    """
    f(x0, x1) = x0² + x1²
    
    최솟값: (0, 0)에서 0
    """
    return x[0]**2 + x[1]**2


def numerical_gradient(f, x):
    """
    수치 미분으로 기울기 계산
    
    Args:
        f: 미분할 함수
        x: 위치 (벡터)
    
    Returns:
        각 위치에서의 기울기
    """
    h = 1e-4  # 작은 값
    grad = np.zeros_like(x)
    
    for idx in range(x.size):
        tmp_val = x[idx]
        
        # f(x + h) 계산
        x[idx] = tmp_val + h
        fxh1 = f(x)
        
        # f(x - h) 계산
        x[idx] = tmp_val - h
        fxh2 = f(x)
        
        # 중앙 차분
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val  # 원래 값 복원
    
    return grad


def gradient_descent(f, init_x, lr=0.1, step_num=100):
    """
    경사하강법
    
    Args:
        f: 최소화할 함수
        init_x: 초기 위치
        lr: 학습률
        step_num: 반복 횟수
    
    Returns:
        최종 위치, 이동 경로
    """
    x = init_x.copy()
    x_history = [x.copy()]
    
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x = x - lr * grad  # 핵심: 기울기 반대 방향으로 이동
        x_history.append(x.copy())
    
    return x, np.array(x_history)


def visualize_gradient_descent():
    """경사하강법 과정 시각화"""
    
    # 초기 위치
    init_x = np.array([-7.0, 2.0])
    
    # 다양한 학습률로 실험
    learning_rates = [0.01, 0.1, 1.0]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, lr in enumerate(learning_rates):
        x_final, x_history = gradient_descent(
            function_2d, init_x.copy(), lr=lr, step_num=20
        )
        
        # 등고선 그리기
        x = np.arange(-10, 10, 0.5)
        y = np.arange(-10, 10, 0.5)
        X, Y = np.meshgrid(x, y)
        Z = X**2 + Y**2
        
        axes[idx].contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.5)
        axes[idx].plot(x_history[:, 0], x_history[:, 1], 'ro-', 
                      markersize=5, linewidth=1.5, alpha=0.7)
        axes[idx].plot(0, 0, 'b*', markersize=20, label='Minimum (0, 0)')
        axes[idx].plot(init_x[0], init_x[1], 'g^', markersize=15, label='Start')
        
        axes[idx].set_xlabel('x0', fontsize=12)
        axes[idx].set_ylabel('x1', fontsize=12)
        axes[idx].set_title(f'Learning Rate = {lr}\n최종 위치: ({x_final[0]:.4f}, {x_final[1]:.4f})', 
                          fontsize=12)
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
        axes[idx].axis('equal')
    
    plt.tight_layout()
    plt.savefig('/home/claude/neural_network_examples/gradient_descent.png', dpi=80)
    plt.close()
    print("그래프 저장됨: gradient_descent.png")


def visualize_3d():
    """3D 시각화"""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 표면 그리기
    x = np.arange(-10, 10, 0.5)
    y = np.arange(-10, 10, 0.5)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2
    
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)
    
    # 경사하강 경로
    init_x = np.array([-7.0, 2.0])
    x_final, x_history = gradient_descent(
        function_2d, init_x.copy(), lr=0.1, step_num=20
    )
    
    z_history = np.array([function_2d(x) for x in x_history])
    ax.plot(x_history[:, 0], x_history[:, 1], z_history, 
            'r.-', markersize=8, linewidth=2, label='Gradient Descent Path')
    
    ax.set_xlabel('x0')
    ax.set_ylabel('x1')
    ax.set_zlabel('f(x0, x1)')
    ax.set_title('Gradient Descent on f(x0, x1) = x0² + x1²', fontsize=14)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('/home/claude/neural_network_examples/gradient_descent_3d.png', dpi=80)
    plt.close()
    print("그래프 저장됨: gradient_descent_3d.png")


def step_by_step_example():
    """단계별 예제"""
    print("\n=== 경사하강법 단계별 예제 ===")
    print("함수: f(x0, x1) = x0² + x1²")
    print("최솟값: (0, 0)에서 0")
    
    init_x = np.array([-3.0, 4.0])
    lr = 0.1
    
    print(f"\n초기 위치: {init_x}")
    print(f"학습률: {lr}")
    print(f"\n반복   현재 위치           기울기              함수값")
    print("-" * 70)
    
    x = init_x.copy()
    for i in range(10):
        grad = numerical_gradient(function_2d, x)
        f_val = function_2d(x)
        print(f"{i:>3}    [{x[0]:>7.4f}, {x[1]:>7.4f}]   "
              f"[{grad[0]:>7.4f}, {grad[1]:>7.4f}]   {f_val:>7.4f}")
        x = x - lr * grad
    
    print(f"\n최종 위치: {x}")
    print(f"최종 함수값: {function_2d(x):.4f}")


if __name__ == "__main__":
    print("=" * 50)
    print("경사하강법 (Gradient Descent) 실습")
    print("=" * 50)
    
    # 단계별 예제
    step_by_step_example()
    
    # 시각화
    print("\n그래프 생성 중...")
    visualize_gradient_descent()
    visualize_3d()
    
    print("\n" + "=" * 50)
    print("핵심 정리:")
    print("- 학습률이 너무 작으면: 학습이 느림")
    print("- 학습률이 너무 크면: 발산 (튕겨나감)")
    print("- 적절한 학습률 선택이 중요!")
    print("=" * 50)
