"""
==========================================
08. 옵티마이저 비교 (Optimizers)
==========================================
강의 페이지 24-29 내용

4가지 옵티마이저 구현 및 비교:
- SGD (Stochastic Gradient Descent)
- Momentum
- AdaGrad
- Adam
"""

import numpy as np
import matplotlib.pyplot as plt


class SGD:
    """확률적 경사하강법 (Stochastic Gradient Descent)"""
    
    def __init__(self, lr=0.01):
        self.lr = lr
    
    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]


class Momentum:
    """Momentum 옵티마이저 - 관성 효과 추가"""
    
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
    
    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
        
        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]


class AdaGrad:
    """AdaGrad 옵티마이저 - 학습률 자동 조절"""
    
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None
    
    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
        
        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


class Adam:
    """Adam 옵티마이저 - Momentum + AdaGrad"""
    
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
    
    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
        
        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)
        
        for key in params.keys():
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])
            
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)


def f(x, y):
    """비등방성 함수: f(x, y) = x²/20 + y²"""
    return x**2 / 20.0 + y**2


def df(x, y):
    """f의 기울기"""
    return x / 10.0, 2.0 * y


def visualize_optimizers():
    """4가지 옵티마이저의 경로 비교"""
    init_pos = (-7.0, 2.0)
    
    optimizers = {
        'SGD': SGD(lr=0.95),
        'Momentum': Momentum(lr=0.1),
        'AdaGrad': AdaGrad(lr=1.5),
        'Adam': Adam(lr=0.3)
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()
    
    for idx, (name, optimizer) in enumerate(optimizers.items()):
        params = {'x': np.array([init_pos[0]]), 'y': np.array([init_pos[1]])}
        grads = {'x': np.array([0.0]), 'y': np.array([0.0])}
        
        x_history = [params['x'][0]]
        y_history = [params['y'][0]]
        
        for i in range(30):
            grads['x'][0], grads['y'][0] = df(params['x'][0], params['y'][0])
            optimizer.update(params, grads)
            
            x_history.append(params['x'][0])
            y_history.append(params['y'][0])
        
        # 등고선 그리기
        x = np.arange(-10, 10, 0.01)
        y = np.arange(-5, 5, 0.01)
        X, Y = np.meshgrid(x, y)
        Z = f(X, Y)
        
        # 마스크
        mask = Z > 7
        Z[mask] = 0
        
        axes[idx].contour(X, Y, Z, levels=10, colors='gray', alpha=0.5)
        axes[idx].plot(x_history, y_history, 'r.-', markersize=8, linewidth=1.5)
        axes[idx].plot(0, 0, 'b+', markersize=20, markeredgewidth=3, label='Minimum')
        axes[idx].plot(init_pos[0], init_pos[1], 'g^', markersize=15, label='Start')
        
        axes[idx].set_title(f'{name}', fontsize=14)
        axes[idx].set_xlim(-10, 10)
        axes[idx].set_ylim(-5, 5)
        axes[idx].set_xlabel('x')
        axes[idx].set_ylabel('y')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].legend()
    
    plt.tight_layout()
    plt.savefig('/home/claude/neural_network_examples/optimizers_comparison.png', dpi=80)
    plt.close()
    print("그래프 저장됨: optimizers_comparison.png")


def compare_on_neural_network():
    """간단한 신경망 학습으로 옵티마이저 비교"""
    print("\n" + "=" * 60)
    print("신경망 학습으로 옵티마이저 비교")
    print("=" * 60)
    
    np.random.seed(42)
    
    # 데이터 생성
    N = 100
    X = np.random.randn(N, 5)
    y = np.random.randn(N, 3)
    
    optimizers = {
        'SGD': SGD(lr=0.01),
        'Momentum': Momentum(lr=0.01),
        'AdaGrad': AdaGrad(lr=0.01),
        'Adam': Adam(lr=0.01)
    }
    
    losses = {name: [] for name in optimizers.keys()}
    
    for name, optimizer in optimizers.items():
        np.random.seed(42)
        params = {
            'W1': np.random.randn(5, 10) * 0.01,
            'W2': np.random.randn(10, 3) * 0.01
        }
        
        for epoch in range(500):
            # 순방향
            h = 1 / (1 + np.exp(-X.dot(params['W1'])))
            y_pred = h.dot(params['W2'])
            loss = 0.5 * np.sum((y_pred - y)**2)
            losses[name].append(loss)
            
            # 역방향
            grad_y_pred = y_pred - y
            grads = {
                'W2': h.T.dot(grad_y_pred),
                'W1': X.T.dot(grad_y_pred.dot(params['W2'].T) * h * (1 - h))
            }
            
            optimizer.update(params, grads)
    
    # 그래프
    plt.figure(figsize=(12, 6))
    for name, loss_history in losses.items():
        plt.plot(loss_history, label=name, linewidth=1.5)
    
    plt.title('Optimizer Comparison on Neural Network', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.yscale('log')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig('/home/claude/neural_network_examples/optimizers_nn.png', dpi=80)
    plt.close()
    print("그래프 저장됨: optimizers_nn.png")
    
    # 최종 손실 출력
    print("\n최종 손실값:")
    for name, loss_history in losses.items():
        print(f"  {name:>10}: {loss_history[-1]:.4f}")


if __name__ == "__main__":
    print("=" * 60)
    print("옵티마이저 비교 실험")
    print("=" * 60)
    
    print("\n각 옵티마이저의 특징:")
    print("- SGD: 가장 기본, 지그재그 경로")
    print("- Momentum: 관성 사용, 더 부드러운 경로")
    print("- AdaGrad: 학습률 자동 조절")
    print("- Adam: 가장 균형 잡힌 성능")
    
    # 시각화
    visualize_optimizers()
    
    # 신경망에 적용
    compare_on_neural_network()
    
    print("\n" + "=" * 60)
    print("핵심 정리:")
    print("- SGD: 가장 단순하지만 느림")
    print("- Momentum: 빠르고 안정적")
    print("- AdaGrad: 자동 학습률 조절")
    print("- Adam: 가장 많이 사용 (추천!)")
    print("=" * 60)
