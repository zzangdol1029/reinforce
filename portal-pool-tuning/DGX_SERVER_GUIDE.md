# DGX Station 서버 실행 가이드

> **서버 사양**: Intel Xeon E5-2698 v4 (20코어/40스레드) · NVIDIA V100 32GB × 4장

---

## 1. 서버에 올릴 파일 목록

아래 파일만 업로드합니다. `__pycache__`, `models/`, `results/` 는 제외합니다.

```
portal-pool-tuning/
├── envs/
│   ├── __init__.py          ✅ 업로드
│   └── portal_env.py        ✅ 업로드  (시뮬레이터 환경)
├── baselines.py             ✅ 업로드  (비교 기준 정책)
├── config.py                ✅ 업로드  (포털 사양 설정)
├── train.py                 ✅ 업로드  (학습 실행)
├── optimize.py              ✅ 업로드  (하이퍼파라미터 탐색)
├── evaluate.py              ✅ 업로드  (결과 평가·시각화)
└── requirements.txt         ✅ 업로드  (패키지 목록)
```

### 파일 전송 명령어 (로컬 → 서버)

```bash
# rsync 권장 (__pycache__ 등 자동 제외)
rsync -avz --exclude='__pycache__' \
           --exclude='*.pyc' \
           --exclude='models/' \
           --exclude='results/' \
           portal-pool-tuning/ \
           <계정>@<서버IP>:~/workspace/portal-pool-tuning/

# scp 사용 시
scp -r portal-pool-tuning/ <계정>@<서버IP>:~/workspace/
```

---

## 2. 서버 환경 설정 (최초 1회)

```bash
# 서버 접속
ssh <계정>@<서버IP>
cd ~/workspace/portal-pool-tuning

# Conda 환경 생성 및 패키지 설치
conda create -n portal-rl python=3.11 -y
conda activate portal-rl
pip install -r requirements.txt

# GPU 인식 확인 (V100 4장 보여야 정상)
nvidia-smi
python -c "import torch; print(torch.cuda.device_count(), 'GPUs available')"
```

---

## 3. 작업 흐름

```
[선택] optimize.py   →   train.py   →   evaluate.py   →   결과 다운로드
  HP 탐색 (~1시간)      학습 (~30분)     평가·그래프 생성      로컬에서 확인
```

---

## 4. 하이퍼파라미터 탐색 (optimize.py)

학습 전에 최적 하이퍼파라미터를 찾는 단계입니다. 생략하고 기본값으로 바로 학습해도 됩니다.

```bash
conda activate portal-rl
cd ~/workspace/portal-pool-tuning
mkdir -p logs

# GPU별 알고리즘 분리 탐색 (동시 실행)
CUDA_VISIBLE_DEVICES=0 nohup python optimize.py --portal research --algo DQN \
    --n-trials 50 --timesteps 80000 > logs/opt_DQN.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python optimize.py --portal research --algo PPO \
    --n-trials 50 --timesteps 80000 > logs/opt_PPO.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python optimize.py --portal research --algo SAC \
    --n-trials 50 --timesteps 80000 > logs/opt_SAC.log 2>&1 &
```

탐색 결과 확인:
```bash
cat results/optuna_research_DQN_best.txt   # 최적 HP 출력
```

> 중단 후 재시작해도 기존 결과를 이어받습니다 (`results/optuna_*.db` 유지).

---

## 5. 학습 실행 (train.py)

### GPU별 포털 분리 동시 실행 (권장)

```bash
mkdir -p logs

CUDA_VISIBLE_DEVICES=0 nohup python train.py --portal research --algo all \
    --timesteps 250000 > logs/train_research.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python train.py --portal user --algo all \
    --timesteps 250000 > logs/train_user.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python train.py --portal admin --algo all \
    --timesteps 250000 > logs/train_admin.log 2>&1 &

# 모든 작업 완료 대기
wait && echo "전체 학습 완료"
```

### 학습 진행 상황 확인

```bash
# 실시간 로그 확인
tail -f logs/train_research.log

# 실행 중인 프로세스 확인
ps aux | grep train.py
```

> **예상 시간**: 3개 포털 동시 실행 기준 약 **30~40분**

---

## 6. 평가 실행 (evaluate.py)

```bash
python evaluate.py --portal research --episodes 20
python evaluate.py --portal user     --episodes 20
python evaluate.py --portal admin    --episodes 20
```

생성 결과물:
```
results/
├── comparison_<portal>.png          ← 보상·SLA·자원 막대 비교 그래프
├── comparison_<portal>.csv          ← 수치 데이터
├── timeseries_<portal>.png          ← W·D vs 부하 시계열
├── learning_curve_<portal>_<algo>.png  ← 학습 곡선 (수렴·과적합 표시)
└── optuna_<portal>_<algo>_history.png  ← HP 탐색 진행 그래프
```

---

## 7. 결과 파일 다운로드 (서버 → 로컬)

```bash
# 로컬 터미널에서 실행
rsync -avz <계정>@<서버IP>:~/workspace/portal-pool-tuning/results/ ./results/

# scp 사용 시
scp -r <계정>@<서버IP>:~/workspace/portal-pool-tuning/results/ ./results/
```

---

## 8. GPU 모니터링

```bash
# 1초마다 GPU 상태 갱신
watch -n 1 nvidia-smi

# GPU별 이용률·메모리 간단 확인
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
           --format=csv -l 2
```

> GPU 이용률이 20% 이하로 낮으면 정상입니다.
> 이 프로젝트는 Python 환경 시뮬레이션이 병목이라 CPU 중심으로 동작합니다.

---

## 9. 자주 발생하는 문제

| 증상 | 원인 | 해결 |
|------|------|------|
| `ModuleNotFoundError` | conda 환경 미활성화 | `conda activate portal-rl` 확인 |
| GPU 이용률 5% 이하 | 환경 시뮬레이션 병목 | 정상 동작 |
| nohup 프로세스 확인 불가 | PID 분실 | `ps aux \| grep python` |
| 탐색 중단 후 재실행 | DB는 자동 저장됨 | 동일 명령어 재실행하면 이어받음 |
| `CUDA out of memory` | 배치 크기 과다 (거의 없음) | `--batch-size` 줄이기 |
