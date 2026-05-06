"""
==========================================
10. 과적합 방지 (Overfitting Prevention)
==========================================
강의 페이지 41-44 내용

과적합을 방지하는 방법:
- Weight Decay (가중치 감소)
- Dropout (뉴런 일부 끄기)
"""

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


class Dropout:
    """
    Dropout 레이어
    
    학습 시: 일부 뉴런을 랜덤하게 끔
    테스트 시: 모든 뉴런 사용 (출력에 비율 곱함)
    """
    
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None
    
    def forward(self, x, train_flg=True):
        if train_flg:
            # 학습 시: 일부를 0으로 만듦
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            # 테스트 시: 비율 곱하기
            return x * (1.0 - self.dropout_ratio)
    
    def backward(self, dout):
        return dout * self.mask


class NeuralNetWithRegularization:
    """과적합 방지 기능을 가진 신경망"""
    
    def __init__(self, input_size, hidden_size, output_size, 
                 weight_decay_lambda=0, use_dropout=False, dropout_ratio=0.15):
        self.params = {}
        self.params['W1'] = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.params['b2'] = np.zeros(output_size)
        
        self.weight_decay_lambda = weight_decay_lambda
        self.use_dropout = use_dropout
        self.dropout = Dropout(dropout_ratio) if use_dropout else None
    
    def predict(self, x, train_flg=False):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        
        a1 = np.dot(x, W1) + b1
        z1 = relu(a1)
        
        # Dropout 적용
        if self.use_dropout:
            z1 = self.dropout.forward(z1, train_flg)
        
        a2 = np.dot(z1, W2) + b2
        # Softmax
        a2 = a2 - np.max(a2, axis=1, keepdims=True)
        y = np.exp(a2) / np.sum(np.exp(a2), axis=1, keepdims=True)
        
        return y
    
    def loss(self, x, t, train_flg=False):
        y = self.predict(x, train_flg)
        delta = 1e-7
        loss = -np.sum(t * np.log(y + delta)) / x.shape[0]
        
        # Weight Decay 추가
        if self.weight_decay_lambda > 0:
            weight_norm = 0
            for key in ['W1', 'W2']:
                weight_norm += np.sum(self.params[key] ** 2)
            loss += 0.5 * self.weight_decay_lambda * weight_norm
        
        return loss
    
    def accuracy(self, x, t):
        y = self.predict(x, train_flg=False)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        return np.sum(y == t) / float(x.shape[0])
    
    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        
        grads = {}
        batch_num = x.shape[0]
        
        # 순방향
        a1 = np.dot(x, W1) + b1
        z1 = relu(a1)
        if self.use_dropout:
            z1 = self.dropout.forward(z1, train_flg=True)
        
        a2 = np.dot(z1, W2) + b2
        a2 = a2 - np.max(a2, axis=1, keepdims=True)
        y = np.exp(a2) / np.sum(np.exp(a2), axis=1, keepdims=True)
        
        # 역방향
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy) + self.weight_decay_lambda * W2
        grads['b2'] = np.sum(dy, axis=0)
        
        da1 = np.dot(dy, W2.T)
        if self.use_dropout:
            da1 = self.dropout.backward(da1)
        dz1 = da1.copy()
        dz1[a1 <= 0] = 0  # ReLU 미분
        
        grads['W1'] = np.dot(x.T, dz1) + self.weight_decay_lambda * W1
        grads['b1'] = np.sum(dz1, axis=0)
        
        return grads


def generate_overfitting_data(num_train=300, num_test=100):
    """과적합을 만들기 쉬운 데이터셋 생성"""
    np.random.seed(42)
    
    num_classes = 5
    feature_dim = 20
    
    # 훈련 데이터 (적은 양)
    X_train = np.random.randn(num_train, feature_dim)
    y_train_labels = np.random.randint(0, num_classes, num_train)
    
    # one-hot encoding
    y_train = np.zeros((num_train, num_classes))
    y_train[np.arange(num_train), y_train_labels] = 1
    
    # 테스트 데이터
    X_test = np.random.randn(num_test, feature_dim)
    y_test_labels = np.random.randint(0, num_classes, num_test)
    y_test = np.zeros((num_test, num_classes))
    y_test[np.arange(num_test), y_test_labels] = 1
    
    return X_train, y_train, X_test, y_test


def train_network(network, X_train, y_train, X_test, y_test, 
                  num_epochs=200, learning_rate=0.01):
    """신경망 학습"""
    train_acc_list = []
    test_acc_list = []
    
    for epoch in range(num_epochs):
        # 학습
        grads = network.gradient(X_train, y_train)
        for key in network.params.keys():
            network.params[key] -= learning_rate * grads[key]
        
        # 정확도 측정
        train_acc = network.accuracy(X_train, y_train)
        test_acc = network.accuracy(X_test, y_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
    
    return train_acc_list, test_acc_list


def compare_methods():
    """과적합 방지 방법 비교"""
    print("=" * 60)
    print("과적합 방지 방법 비교 실험")
    print("=" * 60)
    
    # 데이터 생성
    X_train, y_train, X_test, y_test = generate_overfitting_data()
    
    print(f"\n데이터 설정:")
    print(f"  훈련 데이터: {X_train.shape}")
    print(f"  테스트 데이터: {X_test.shape}")
    
    # 1. 일반 신경망 (과적합 발생)
    print("\n[1] 일반 신경망 (과적합 발생 예상)")
    np.random.seed(42)
    net1 = NeuralNetWithRegularization(input_size=20, hidden_size=100, output_size=5)
    train_acc1, test_acc1 = train_network(net1, X_train, y_train, X_test, y_test)
    print(f"   최종 훈련 정확도: {train_acc1[-1]:.2%}")
    print(f"   최종 테스트 정확도: {test_acc1[-1]:.2%}")
    print(f"   차이: {(train_acc1[-1] - test_acc1[-1]):.2%}")
    
    # 2. Weight Decay 적용
    print("\n[2] Weight Decay 적용 (λ=0.1)")
    np.random.seed(42)
    net2 = NeuralNetWithRegularization(input_size=20, hidden_size=100, output_size=5,
                                        weight_decay_lambda=0.1)
    train_acc2, test_acc2 = train_network(net2, X_train, y_train, X_test, y_test)
    print(f"   최종 훈련 정확도: {train_acc2[-1]:.2%}")
    print(f"   최종 테스트 정확도: {test_acc2[-1]:.2%}")
    print(f"   차이: {(train_acc2[-1] - test_acc2[-1]):.2%}")
    
    # 3. Dropout 적용
    print("\n[3] Dropout 적용 (ratio=0.15)")
    np.random.seed(42)
    net3 = NeuralNetWithRegularization(input_size=20, hidden_size=100, output_size=5,
                                        use_dropout=True, dropout_ratio=0.15)
    train_acc3, test_acc3 = train_network(net3, X_train, y_train, X_test, y_test)
    print(f"   최종 훈련 정확도: {train_acc3[-1]:.2%}")
    print(f"   최종 테스트 정확도: {test_acc3[-1]:.2%}")
    print(f"   차이: {(train_acc3[-1] - test_acc3[-1]):.2%}")
    
    # 시각화
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].plot(train_acc1, label='Training', linewidth=2)
    axes[0].plot(test_acc1, label='Test', linewidth=2)
    axes[0].set_title('No Regularization (Overfitting)', fontsize=14)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1.0)
    
    axes[1].plot(train_acc2, label='Training', linewidth=2)
    axes[1].plot(test_acc2, label='Test', linewidth=2)
    axes[1].set_title('Weight Decay (λ=0.1)', fontsize=14)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1.0)
    
    axes[2].plot(train_acc3, label='Training', linewidth=2)
    axes[2].plot(test_acc3, label='Test', linewidth=2)
    axes[2].set_title('Dropout (ratio=0.15)', fontsize=14)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Accuracy')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig('/home/claude/neural_network_examples/regularization.png', dpi=80)
    plt.close()
    print("\n그래프 저장됨: regularization.png")


def dropout_demo():
    """Dropout 동작 시연"""
    print("\n" + "=" * 60)
    print("Dropout 동작 시연")
    print("=" * 60)
    
    np.random.seed(42)
    dropout = Dropout(dropout_ratio=0.5)
    
    # 입력
    x = np.array([[1.0, 2.0, 3.0, 4.0, 5.0],
                  [6.0, 7.0, 8.0, 9.0, 10.0]])
    
    print(f"\n입력 x:\n{x}")
    
    # 학습 시
    print("\n[학습 시] - 50%를 랜덤하게 끔")
    for i in range(3):
        np.random.seed(i)
        out = dropout.forward(x, train_flg=True)
        print(f"\n시도 {i+1}:")
        print(f"마스크:\n{dropout.mask.astype(int)}")
        print(f"출력:\n{out}")
    
    # 테스트 시
    print("\n[테스트 시] - 모든 뉴런 사용 (50%만큼 곱함)")
    out_test = dropout.forward(x, train_flg=False)
    print(f"출력:\n{out_test}")


if __name__ == "__main__":
    # Dropout 시연
    dropout_demo()
    
    # 과적합 방지 방법 비교
    compare_methods()
    
    print("\n" + "=" * 60)
    print("핵심 정리:")
    print("- 과적합: 훈련 데이터만 외우고 새 데이터는 못 맞춤")
    print("- Weight Decay: 가중치를 작게 유지")
    print("- Dropout: 뉴런 일부를 랜덤하게 끔")
    print("- 두 방법 모두 일반화 성능 향상!")
    print("=" * 60)
