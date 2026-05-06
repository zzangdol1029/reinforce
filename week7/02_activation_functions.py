"""
==========================================
02. 활성화 함수 (Activation Functions)
==========================================
강의 페이지 5, 33 내용

다양한 활성화 함수를 구현하고 시각화합니다.
- Step (계단 함수)
- Sigmoid
- Tanh
- ReLU
- Leaky ReLU
- ELU
"""

import numpy as np
import matplotlib.pyplot as plt


def step_function(x):
    """계단 함수: 0보다 크면 1, 작으면 0"""
    return np.where(x > 0, 1, 0)


def sigmoid(x):
    """시그모이드 함수: 0~1 사이의 부드러운 곡선"""
    return 1 / (1 + np.exp(-x))


def tanh(x):
    """하이퍼볼릭 탄젠트: -1~1 사이의 곡선"""
    return np.tanh(x)


def relu(x):
    """ReLU: 0보다 크면 그대로, 작으면 0"""
    return np.maximum(0, x)


def leaky_relu(x, alpha=0.1):
    """Leaky ReLU: 음수 영역도 약간 살림"""
    return np.where(x > 0, x, alpha * x)


def elu(x, alpha=1.0):
    """Exponential Linear Unit: 부드러운 ReLU"""
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))


def softmax(x):
    """
    Softmax 함수: 확률로 변환
    분류 문제의 출력층에서 사용
    """
    # 오버플로우 방지를 위해 최댓값을 빼줌
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)


def plot_activation_functions():
    """모든 활성화 함수를 한 그림에 그리기"""
    x = np.linspace(-5, 5, 100)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Step 함수
    axes[0, 0].plot(x, step_function(x), 'b-', linewidth=2)
    axes[0, 0].set_title('Step Function', fontsize=14)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0, color='k', linewidth=0.5)
    axes[0, 0].axvline(x=0, color='k', linewidth=0.5)
    
    # Sigmoid 함수
    axes[0, 1].plot(x, sigmoid(x), 'r-', linewidth=2)
    axes[0, 1].set_title('Sigmoid Function', fontsize=14)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0, color='k', linewidth=0.5)
    axes[0, 1].axvline(x=0, color='k', linewidth=0.5)
    
    # Tanh 함수
    axes[0, 2].plot(x, tanh(x), 'g-', linewidth=2)
    axes[0, 2].set_title('Tanh Function', fontsize=14)
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].axhline(y=0, color='k', linewidth=0.5)
    axes[0, 2].axvline(x=0, color='k', linewidth=0.5)
    
    # ReLU 함수
    axes[1, 0].plot(x, relu(x), 'm-', linewidth=2)
    axes[1, 0].set_title('ReLU Function', fontsize=14)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=0, color='k', linewidth=0.5)
    axes[1, 0].axvline(x=0, color='k', linewidth=0.5)
    
    # Leaky ReLU 함수
    axes[1, 1].plot(x, leaky_relu(x), 'c-', linewidth=2)
    axes[1, 1].set_title('Leaky ReLU Function', fontsize=14)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=0, color='k', linewidth=0.5)
    axes[1, 1].axvline(x=0, color='k', linewidth=0.5)
    
    # ELU 함수
    axes[1, 2].plot(x, elu(x), 'y-', linewidth=2)
    axes[1, 2].set_title('ELU Function', fontsize=14)
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].axhline(y=0, color='k', linewidth=0.5)
    axes[1, 2].axvline(x=0, color='k', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('/home/claude/neural_network_examples/activation_functions.png', dpi=80)
    plt.close()
    print("그래프 저장됨: activation_functions.png")


def compare_activations():
    """모든 활성화 함수를 한 그래프에 비교"""
    x = np.linspace(-5, 5, 100)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, sigmoid(x), label='Sigmoid', linewidth=2)
    plt.plot(x, tanh(x), label='Tanh', linewidth=2)
    plt.plot(x, relu(x), label='ReLU', linewidth=2)
    plt.plot(x, leaky_relu(x), label='Leaky ReLU', linewidth=2, linestyle='--')
    plt.plot(x, elu(x), label='ELU', linewidth=2, linestyle=':')
    
    plt.title('Activation Functions Comparison', fontsize=14)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('f(x)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linewidth=0.5)
    plt.axvline(x=0, color='k', linewidth=0.5)
    plt.savefig('/home/claude/neural_network_examples/activation_comparison.png', dpi=80)
    plt.close()
    print("그래프 저장됨: activation_comparison.png")


def softmax_example():
    """Softmax 예제: 분류 문제의 확률 변환"""
    print("\n=== Softmax 예제 ===")
    
    # 신경망 출력값 (정규화 전)
    scores = np.array([3.0, 1.0, 0.2])
    print(f"신경망 출력: {scores}")
    
    # Softmax 적용
    probabilities = softmax(scores)
    print(f"Softmax 결과: {probabilities}")
    print(f"확률 합계: {np.sum(probabilities):.4f}")
    
    # 클래스별 확률
    classes = ['강아지', '고양이', '토끼']
    print("\n각 클래스 확률:")
    for cls, prob in zip(classes, probabilities):
        print(f"  {cls}: {prob*100:.2f}%")
    
    # 가장 높은 확률의 클래스 선택
    predicted_class = classes[np.argmax(probabilities)]
    print(f"\n예측: {predicted_class}")


if __name__ == "__main__":
    print("=" * 50)
    print("활성화 함수 시각화 및 실험")
    print("=" * 50)
    
    # 각 활성화 함수의 출력값 예시
    x_test = np.array([-2, -1, 0, 1, 2])
    print(f"\n입력값: {x_test}")
    print(f"Step:       {step_function(x_test)}")
    print(f"Sigmoid:    {sigmoid(x_test).round(4)}")
    print(f"Tanh:       {tanh(x_test).round(4)}")
    print(f"ReLU:       {relu(x_test)}")
    print(f"Leaky ReLU: {leaky_relu(x_test).round(4)}")
    print(f"ELU:        {elu(x_test).round(4)}")
    
    # 그래프 그리기
    plot_activation_functions()
    compare_activations()
    
    # Softmax 예제
    softmax_example()
    
    print("\n" + "=" * 50)
    print("핵심 정리:")
    print("- Step: 켜짐/꺼짐만 (전등 스위치)")
    print("- Sigmoid: 0~1 부드러운 곡선 (출력층용)")
    print("- Tanh: -1~1 곡선")
    print("- ReLU: 가장 많이 사용 (계산 빠름)")
    print("- Softmax: 다중 분류의 확률 변환")
    print("=" * 50)
