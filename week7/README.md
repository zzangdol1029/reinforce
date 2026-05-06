# 🧠 신경망 실습 코드 모음

신경망 강의 자료의 핵심 개념을 직접 코드로 구현한 실습 예제입니다.

## 📁 파일 구성

| 파일명 | 내용 | 강의 페이지 |
|--------|------|------------|
| `01_perceptron.py` | 퍼셉트론으로 AND/OR/NAND/XOR 게이트 구현 | 페이지 2-4 |
| `02_activation_functions.py` | 활성화 함수 (Step, Sigmoid, Tanh, ReLU 등) | 페이지 5, 33 |
| `03_loss_functions.py` | 손실 함수 (MSE, CEE) | 페이지 10 |
| `04_gradient_descent.py` | 경사하강법 구현 및 시각화 | 페이지 12 |
| `05_perceptron_learning.py` | 퍼셉트론 학습 알고리즘 | 페이지 13-14 |
| `06_backpropagation.py` | 역전파 알고리즘 단계별 구현 | 페이지 15-18 |
| `07_two_layer_network.py` | 2층 신경망 (20줄 코드 + XOR 학습) | 페이지 19 |
| `08_optimizers.py` | SGD, Momentum, AdaGrad, Adam 비교 | 페이지 24-29 |
| `09_weight_initialization.py` | Xavier, He 초기화 비교 | 페이지 34-38 |
| `10_overfitting_prevention.py` | Weight Decay, Dropout | 페이지 41-44 |

## 🚀 실행 방법

### 필요한 라이브러리 설치

```bash
pip install numpy matplotlib
```

### 개별 실행

```bash
# 각 파일을 개별적으로 실행
python 01_perceptron.py
python 02_activation_functions.py
python 03_loss_functions.py
# ... 등등
```

### 전체 실행

```bash
# 모든 예제 한 번에 실행
for f in *.py; do
    echo "===== Running $f ====="
    python "$f"
done
```

## 📊 주요 실습 내용

### 1️⃣ 퍼셉트론 (01_perceptron.py)
- AND, OR, NAND 게이트 구현
- XOR은 다층 퍼셉트론으로 해결
- 결정 경계 시각화

### 2️⃣ 활성화 함수 (02_activation_functions.py)
- 6가지 활성화 함수 구현
- 그래프 비교
- Softmax 분류 예제

### 3️⃣ 손실 함수 (03_loss_functions.py)
- MSE: 회귀 문제용
- CEE: 분류 문제용
- 강아지/고양이/토끼 분류 예제

### 4️⃣ 경사하강법 (04_gradient_descent.py)
- 2D 함수 최적화
- 학습률에 따른 수렴 비교
- 3D 시각화

### 5️⃣ 퍼셉트론 학습 (05_perceptron_learning.py)
- 강의 예제 단계별 풀이
- AND 게이트 자동 학습
- 결정 경계 시각화

### 6️⃣ 역전파 (06_backpropagation.py)
- 강의 예제 (페이지 17-18) 단계별 구현
- 순방향/역방향 계산
- 가중치 업데이트

### 7️⃣ 2층 신경망 (07_two_layer_network.py)
- 강의의 20줄 코드 그대로 구현
- 객체지향 버전
- XOR 문제 해결

### 8️⃣ 옵티마이저 (08_optimizers.py)
- 4가지 옵티마이저 직접 구현
- 비등방성 함수에서 경로 비교
- 신경망 학습 성능 비교

### 9️⃣ 가중치 초기화 (09_weight_initialization.py)
- Sigmoid + 표준편차 1: 기울기 소실
- Sigmoid + 표준편차 0.01: 표현력 제한
- Xavier 초기화: Sigmoid 적합
- He 초기화: ReLU 적합

### 🔟 과적합 방지 (10_overfitting_prevention.py)
- 과적합 발생 시뮬레이션
- Weight Decay (L2 정규화)
- Dropout 동작 시연

## 📈 생성되는 그래프

각 코드 실행 시 다음과 같은 PNG 파일이 생성됩니다:

```
gate_AND.png, gate_OR.png, gate_NAND.png, gate_XOR.png
activation_functions.png, activation_comparison.png
loss_functions.png
gradient_descent.png, gradient_descent_3d.png
perceptron_learning.png, perceptron_boundary.png
backpropagation_curve.png
simple_2layer.png, xor_learning.png, xor_boundary.png
optimizers_comparison.png, optimizers_nn.png
init_std1.png, init_std001.png, init_xavier.png
init_relu_std001.png, init_relu_xavier.png, init_relu_he.png
init_training_speed.png
regularization.png
```

## 💡 학습 순서 권장

1. **기본 개념**: `01_perceptron.py` → `02_activation_functions.py`
2. **학습 원리**: `03_loss_functions.py` → `04_gradient_descent.py`
3. **알고리즘**: `05_perceptron_learning.py` → `06_backpropagation.py`
4. **실전 구현**: `07_two_layer_network.py`
5. **최적화 기법**: `08_optimizers.py` → `09_weight_initialization.py`
6. **고급 기법**: `10_overfitting_prevention.py`

## 🎯 학습 목표

이 실습을 모두 완료하면:
- ✅ 신경망의 기본 원리를 이해
- ✅ 순방향/역방향 계산 가능
- ✅ 학습 과정을 시각적으로 이해
- ✅ 다양한 최적화 기법 비교 가능
- ✅ 과적합 문제와 해결 방법 숙지

## 📚 참고 자료

- 강의: 박태형 교수 (충북대학교 지능시스템 및 로봇공학과)
- 도서: 「밑바닥부터 시작하는 딥러닝」 (사이토 고키)

---

**🚀 즐거운 신경망 학습 되세요!**
