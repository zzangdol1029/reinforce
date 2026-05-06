"""
==========================================
01. 퍼셉트론 (Perceptron)
==========================================
강의 페이지 2-3 내용

단일 퍼셉트론으로 OR, AND, NAND 게이트를 구현합니다.
XOR은 단일 퍼셉트론으로 구현 불가능함을 확인합니다.
"""

import numpy as np
import matplotlib.pyplot as plt


def perceptron(x1, x2, w1, w2, b):
    """
    기본 퍼셉트론 함수
    
    Args:
        x1, x2: 입력값
        w1, w2: 가중치
        b: 편향(bias)
    
    Returns:
        0 또는 1
    """
    y = w1 * x1 + w2 * x2 + b
    if y <= 0:
        return 0
    else:
        return 1


def AND(x1, x2):
    """AND 게이트: 둘 다 1이어야 1"""
    w1, w2, b = 0.5, 0.5, -0.7
    return perceptron(x1, x2, w1, w2, b)


def OR(x1, x2):
    """OR 게이트: 하나라도 1이면 1"""
    w1, w2, b = 1.0, 1.0, -0.5
    return perceptron(x1, x2, w1, w2, b)


def NAND(x1, x2):
    """NAND 게이트: AND의 반대"""
    w1, w2, b = -0.5, -0.5, 0.7
    return perceptron(x1, x2, w1, w2, b)


def XOR(x1, x2):
    """
    XOR 게이트: 다층 퍼셉트론(2층)으로 구현
    단일 퍼셉트론으로는 불가능!
    """
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y


def test_gate(gate_func, gate_name):
    """게이트 테스트 함수"""
    print(f"\n=== {gate_name} 게이트 테스트 ===")
    print(f"{'x1':>3} {'x2':>3} | {'y':>3}")
    print("-" * 15)
    for x1 in [0, 1]:
        for x2 in [0, 1]:
            y = gate_func(x1, x2)
            print(f"{x1:>3} {x2:>3} | {y:>3}")


def visualize_gate(gate_func, gate_name):
    """게이트의 결정 경계를 시각화"""
    # 격자점 생성
    x1_range = np.linspace(-0.5, 1.5, 100)
    x2_range = np.linspace(-0.5, 1.5, 100)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    
    # 각 점에 대해 게이트 출력 계산
    Y = np.zeros_like(X1)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            Y[i, j] = gate_func(X1[i, j], X2[i, j])
    
    # 그래프 그리기
    plt.figure(figsize=(6, 6))
    plt.contourf(X1, X2, Y, levels=[-0.5, 0.5, 1.5], 
                 colors=['lightblue', 'lightyellow'], alpha=0.5)
    
    # 진리표의 점들 표시
    for x1 in [0, 1]:
        for x2 in [0, 1]:
            y = gate_func(x1, x2)
            color = 'red' if y == 1 else 'blue'
            marker = 'o' if y == 1 else 'x'
            plt.scatter(x1, x2, c=color, marker=marker, s=200, 
                       edgecolors='black', linewidths=2,
                       label=f'y={y}' if (x1, x2) in [(0, 0), (1, 1)] else '')
    
    plt.xlabel('x1', fontsize=12)
    plt.ylabel('x2', fontsize=12)
    plt.title(f'{gate_name} Gate', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xlim(-0.5, 1.5)
    plt.ylim(-0.5, 1.5)
    plt.savefig(f'/home/claude/neural_network_examples/gate_{gate_name}.png', dpi=80)
    plt.close()
    print(f"  → 그래프 저장됨: gate_{gate_name}.png")


if __name__ == "__main__":
    print("=" * 50)
    print("퍼셉트론으로 논리 게이트 구현하기")
    print("=" * 50)
    
    # 각 게이트 테스트
    test_gate(AND, "AND")
    test_gate(OR, "OR")
    test_gate(NAND, "NAND")
    test_gate(XOR, "XOR (2층 퍼셉트론)")
    
    # 시각화
    print("\n그래프 생성 중...")
    visualize_gate(AND, "AND")
    visualize_gate(OR, "OR")
    visualize_gate(NAND, "NAND")
    visualize_gate(XOR, "XOR")
    
    print("\n" + "=" * 50)
    print("핵심 정리:")
    print("- AND, OR, NAND: 단일 퍼셉트론으로 구현 가능 (선형 분리)")
    print("- XOR: 다층 퍼셉트론 필요 (비선형 분리)")
    print("=" * 50)
