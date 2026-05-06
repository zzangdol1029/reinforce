"""
==========================================
09. 가중치 초기화 (Weight Initialization)
==========================================
강의 페이지 34-38 내용

가중치 초기화 방법에 따른 은닉층 출력 분포 비교:
- 표준편차 1 (잘못된 예)
- 표준편차 0.01 (잘못된 예)
- Xavier 초기화 (Sigmoid/tanh용)
- He 초기화 (ReLU용)
"""

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def experiment_weight_init(weight_std, activation='sigmoid', node_num=100, 
                          hidden_layer_size=5, num_data=1000):
    """
    가중치 초기화 실험
    
    Args:
        weight_std: 가중치 표준편차 또는 'xavier', 'he'
        activation: 활성화 함수
        node_num: 각 층의 노드 수
        hidden_layer_size: 은닉층 수
        num_data: 데이터 개수
    
    Returns:
        각 층의 활성화 값 분포
    """
    x = np.random.randn(num_data, node_num)
    activations = {}
    
    for i in range(hidden_layer_size):
        if i != 0:
            x = activations[i-1]
        
        # 가중치 초기화
        if weight_std == 'xavier':
            w = np.random.randn(node_num, node_num) / np.sqrt(node_num)
        elif weight_std == 'he':
            w = np.random.randn(node_num, node_num) * np.sqrt(2.0 / node_num)
        else:
            w = np.random.randn(node_num, node_num) * weight_std
        
        a = np.dot(x, w)
        
        # 활성화 함수
        if activation == 'sigmoid':
            z = sigmoid(a)
        elif activation == 'relu':
            z = relu(a)
        
        activations[i] = z
    
    return activations


def plot_activation_distribution(activations, title, color='blue'):
    """활성화 값 분포 히스토그램"""
    fig, axes = plt.subplots(1, len(activations), figsize=(20, 4))
    
    for i, a in activations.items():
        axes[i].hist(a.flatten(), 30, range=(0, 1), color=color, alpha=0.7)
        axes[i].set_title(f"Layer {i+1}")
        axes[i].set_xlim(0, 1)
        axes[i].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    return fig


def compare_initialization_sigmoid():
    """Sigmoid 활성화 함수에서 가중치 초기화 비교"""
    print("=" * 60)
    print("Sigmoid 활성화 함수 - 가중치 초기화 비교")
    print("=" * 60)
    
    np.random.seed(42)
    
    # 1. 표준편차 1 (잘못됨)
    print("\n1. 표준편차 = 1 (잘못된 초기화)")
    activations = experiment_weight_init(1.0, 'sigmoid')
    fig = plot_activation_distribution(activations, 
                                       'Sigmoid + std=1 (Bad: Gradient Vanishing)',
                                       color='red')
    fig.savefig('/home/claude/neural_network_examples/init_std1.png', dpi=80)
    plt.close(fig)
    print("   → 0과 1에 집중되어 기울기 소실 발생!")
    
    # 2. 표준편차 0.01 (잘못됨)
    print("\n2. 표준편차 = 0.01 (잘못된 초기화)")
    activations = experiment_weight_init(0.01, 'sigmoid')
    fig = plot_activation_distribution(activations,
                                       'Sigmoid + std=0.01 (Bad: All similar values)',
                                       color='orange')
    fig.savefig('/home/claude/neural_network_examples/init_std001.png', dpi=80)
    plt.close(fig)
    print("   → 0.5에 집중되어 표현력 제한!")
    
    # 3. Xavier 초기화 (좋음)
    print("\n3. Xavier 초기화 (Sigmoid/tanh에 적합)")
    activations = experiment_weight_init('xavier', 'sigmoid')
    fig = plot_activation_distribution(activations,
                                       'Sigmoid + Xavier (Good: Well-distributed)',
                                       color='green')
    fig.savefig('/home/claude/neural_network_examples/init_xavier.png', dpi=80)
    plt.close(fig)
    print("   → 골고루 분포되어 학습이 잘 됨!")
    
    print("\n그래프 저장됨: init_std1.png, init_std001.png, init_xavier.png")


def compare_initialization_relu():
    """ReLU 활성화 함수에서 가중치 초기화 비교"""
    print("\n" + "=" * 60)
    print("ReLU 활성화 함수 - 가중치 초기화 비교")
    print("=" * 60)
    
    np.random.seed(42)
    
    # 1. 표준편차 0.01
    print("\n1. 표준편차 = 0.01")
    activations = experiment_weight_init(0.01, 'relu')
    fig, axes = plt.subplots(1, len(activations), figsize=(20, 4))
    for i, a in activations.items():
        axes[i].hist(a.flatten(), 30, range=(0, 2), color='red', alpha=0.7)
        axes[i].set_title(f"Layer {i+1}")
    plt.suptitle('ReLU + std=0.01 (Bad)', fontsize=14)
    plt.tight_layout()
    plt.savefig('/home/claude/neural_network_examples/init_relu_std001.png', dpi=80)
    plt.close()
    print("   → 깊은 층으로 갈수록 0에 수렴!")
    
    # 2. Xavier 초기화
    print("\n2. Xavier 초기화 (ReLU에는 부적합)")
    activations = experiment_weight_init('xavier', 'relu')
    fig, axes = plt.subplots(1, len(activations), figsize=(20, 4))
    for i, a in activations.items():
        axes[i].hist(a.flatten(), 30, range=(0, 2), color='orange', alpha=0.7)
        axes[i].set_title(f"Layer {i+1}")
    plt.suptitle('ReLU + Xavier (Suboptimal)', fontsize=14)
    plt.tight_layout()
    plt.savefig('/home/claude/neural_network_examples/init_relu_xavier.png', dpi=80)
    plt.close()
    print("   → 깊어질수록 한쪽으로 쏠림")
    
    # 3. He 초기화 (ReLU에 적합)
    print("\n3. He 초기화 (ReLU에 적합)")
    activations = experiment_weight_init('he', 'relu')
    fig, axes = plt.subplots(1, len(activations), figsize=(20, 4))
    for i, a in activations.items():
        axes[i].hist(a.flatten(), 30, range=(0, 2), color='green', alpha=0.7)
        axes[i].set_title(f"Layer {i+1}")
    plt.suptitle('ReLU + He (Good)', fontsize=14)
    plt.tight_layout()
    plt.savefig('/home/claude/neural_network_examples/init_relu_he.png', dpi=80)
    plt.close()
    print("   → 모든 층에서 골고루 분포!")
    
    print("\n그래프 저장됨: init_relu_*.png")


def compare_training_speed():
    """학습 속도 비교 (간단한 실험)"""
    print("\n" + "=" * 60)
    print("학습 속도 비교 (간단 실험)")
    print("=" * 60)
    
    np.random.seed(42)
    
    # 데이터
    N, D_in, H, D_out = 64, 100, 50, 10
    x = np.random.randn(N, D_in)
    y = np.random.randn(N, D_out)
    
    methods = {
        'std=0.01': lambda: (np.random.randn(D_in, H) * 0.01,
                            np.random.randn(H, D_out) * 0.01),
        'Xavier': lambda: (np.random.randn(D_in, H) / np.sqrt(D_in),
                          np.random.randn(H, D_out) / np.sqrt(H)),
        'He': lambda: (np.random.randn(D_in, H) * np.sqrt(2.0/D_in),
                      np.random.randn(H, D_out) * np.sqrt(2.0/H))
    }
    
    losses = {}
    
    for name, init_func in methods.items():
        np.random.seed(42)
        w1, w2 = init_func()
        loss_history = []
        
        for t in range(2000):
            # 순방향 (ReLU)
            h = np.maximum(0, x.dot(w1))
            y_pred = h.dot(w2)
            loss = np.square(y_pred - y).sum()
            loss_history.append(loss)
            
            # 역방향
            grad_y_pred = 2.0 * (y_pred - y)
            grad_w2 = h.T.dot(grad_y_pred)
            grad_h = grad_y_pred.dot(w2.T)
            grad_h[h <= 0] = 0  # ReLU의 미분
            grad_w1 = x.T.dot(grad_h)
            
            w1 -= 1e-4 * grad_w1
            w2 -= 1e-4 * grad_w2
        
        losses[name] = loss_history
    
    # 그래프
    plt.figure(figsize=(10, 6))
    for name, loss_history in losses.items():
        plt.plot(loss_history, label=name, linewidth=1.5)
    
    plt.title('Weight Initialization Effect on Learning Speed (ReLU)', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.yscale('log')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig('/home/claude/neural_network_examples/init_training_speed.png', dpi=80)
    plt.close()
    print("그래프 저장됨: init_training_speed.png")
    
    print("\n최종 손실값:")
    for name, loss_history in losses.items():
        print(f"  {name:>10}: {loss_history[-1]:.4f}")


if __name__ == "__main__":
    print("=" * 60)
    print("가중치 초기화 실험")
    print("=" * 60)
    
    compare_initialization_sigmoid()
    compare_initialization_relu()
    compare_training_speed()
    
    print("\n" + "=" * 60)
    print("핵심 정리:")
    print("- 표준편차 1: 기울기 소실")
    print("- 표준편차 0.01: 표현력 제한")
    print("- Xavier 초기화: Sigmoid, tanh에 적합")
    print("- He 초기화: ReLU에 적합")
    print("- 적절한 초기화가 학습 속도에 큰 영향!")
    print("=" * 60)
