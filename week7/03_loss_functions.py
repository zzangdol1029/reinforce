"""
==========================================
03. 손실 함수 (Loss Functions)
==========================================
강의 페이지 10 내용

신경망 학습의 핵심: 예측값과 정답의 차이를 측정합니다.
- MSE (평균 제곱 오차): 회귀 문제
- CEE (교차 엔트로피 오차): 분류 문제
"""

import numpy as np
import matplotlib.pyplot as plt


def mean_squared_error(y, t):
    """
    평균 제곱 오차 (Mean Squared Error)
    
    E = 1/2 * Σ(y - t)²
    
    회귀 문제(숫자 예측)에 주로 사용
    
    Args:
        y: 신경망의 출력값
        t: 정답 레이블
    
    Returns:
        MSE 값
    """
    return 0.5 * np.sum((y - t) ** 2)


def cross_entropy_error(y, t):
    """
    교차 엔트로피 오차 (Cross Entropy Error)
    
    E = -Σ(t * log(y))
    
    분류 문제(종류 분류)에 주로 사용
    
    Args:
        y: 신경망의 출력값 (확률)
        t: 정답 레이블 (one-hot encoding)
    
    Returns:
        CEE 값
    """
    # log(0) 방지를 위한 작은 값 추가
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))


def cross_entropy_error_batch(y, t):
    """
    미니 배치용 교차 엔트로피 오차
    
    E = -1/N * Σ Σ(t * log(y))
    
    Args:
        y: 신경망 출력 (batch_size, num_classes)
        t: 정답 레이블 (batch_size, num_classes)
    
    Returns:
        배치의 평균 CEE
    """
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    
    batch_size = y.shape[0]
    delta = 1e-7
    return -np.sum(t * np.log(y + delta)) / batch_size


def visualize_loss_functions():
    """손실 함수 시각화"""
    
    # MSE 시각화 (정답 = 1.0)
    y_pred = np.linspace(0, 2, 100)
    t = 1.0
    mse_values = 0.5 * (y_pred - t) ** 2
    
    # CEE 시각화 (정답 = 1)
    y_prob = np.linspace(0.01, 1.0, 100)
    cee_values = -np.log(y_prob)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # MSE 그래프
    axes[0].plot(y_pred, mse_values, 'b-', linewidth=2)
    axes[0].axvline(x=1.0, color='r', linestyle='--', label='True value (t=1.0)')
    axes[0].set_xlabel('Predicted value (y)', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('MSE: Mean Squared Error', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # CEE 그래프
    axes[1].plot(y_prob, cee_values, 'r-', linewidth=2)
    axes[1].axvline(x=1.0, color='b', linestyle='--', label='True value (probability=1)')
    axes[1].set_xlabel('Predicted probability (y)', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_title('CEE: Cross Entropy Error', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/claude/neural_network_examples/loss_functions.png', dpi=80)
    plt.close()
    print("그래프 저장됨: loss_functions.png")


def example_classification():
    """분류 문제 예제"""
    print("\n=== 분류 문제 예제 ===")
    print("문제: 강아지/고양이/토끼 중 무엇인가?")
    
    # 정답: 강아지 (one-hot encoding)
    t = np.array([1, 0, 0])
    
    # 케이스 1: 정확하게 예측 (강아지 90%)
    y1 = np.array([0.9, 0.05, 0.05])
    cee1 = cross_entropy_error(y1, t)
    print(f"\n[케이스 1] 정확한 예측")
    print(f"  예측: 강아지={y1[0]*100:.0f}%, 고양이={y1[1]*100:.0f}%, 토끼={y1[2]*100:.0f}%")
    print(f"  CEE: {cee1:.4f} (낮을수록 좋음)")
    
    # 케이스 2: 애매한 예측
    y2 = np.array([0.4, 0.3, 0.3])
    cee2 = cross_entropy_error(y2, t)
    print(f"\n[케이스 2] 애매한 예측")
    print(f"  예측: 강아지={y2[0]*100:.0f}%, 고양이={y2[1]*100:.0f}%, 토끼={y2[2]*100:.0f}%")
    print(f"  CEE: {cee2:.4f}")
    
    # 케이스 3: 틀린 예측
    y3 = np.array([0.1, 0.8, 0.1])
    cee3 = cross_entropy_error(y3, t)
    print(f"\n[케이스 3] 틀린 예측")
    print(f"  예측: 강아지={y3[0]*100:.0f}%, 고양이={y3[1]*100:.0f}%, 토끼={y3[2]*100:.0f}%")
    print(f"  CEE: {cee3:.4f}")


def example_regression():
    """회귀 문제 예제"""
    print("\n=== 회귀 문제 예제 ===")
    print("문제: 내일 기온 예측 (정답: 25도)")
    
    t = np.array([25.0])
    
    # 케이스 1: 정확한 예측
    y1 = np.array([24.5])
    mse1 = mean_squared_error(y1, t)
    print(f"\n[케이스 1] 예측: {y1[0]:.1f}도, MSE: {mse1:.4f}")
    
    # 케이스 2: 약간 빗나감
    y2 = np.array([23.0])
    mse2 = mean_squared_error(y2, t)
    print(f"[케이스 2] 예측: {y2[0]:.1f}도, MSE: {mse2:.4f}")
    
    # 케이스 3: 많이 빗나감
    y3 = np.array([18.0])
    mse3 = mean_squared_error(y3, t)
    print(f"[케이스 3] 예측: {y3[0]:.1f}도, MSE: {mse3:.4f}")


if __name__ == "__main__":
    print("=" * 50)
    print("손실 함수 (Loss Functions) 실습")
    print("=" * 50)
    
    # 분류 문제 예제
    example_classification()
    
    # 회귀 문제 예제
    example_regression()
    
    # 시각화
    print("\n그래프 생성 중...")
    visualize_loss_functions()
    
    print("\n" + "=" * 50)
    print("핵심 정리:")
    print("- MSE: 회귀 문제 (숫자 예측)")
    print("- CEE: 분류 문제 (종류 분류)")
    print("- 손실값이 낮을수록 좋은 예측")
    print("- 학습은 손실을 최소화하는 과정")
    print("=" * 50)
