# ADMM 기반 전역 가중치 제거를 통한 딥러닝 모델의 압축

- 딥러닝 모델이 높은 성능을 달성하기 위해서는 모델의 가중치 수가 많아진다는 문제점 존재
- Model Compression은 메모리를 절약하고 모델의 저장 크기를 감소시키며 계산 요구사항을 줄일 수 있어 매우 유용
- Model Compression의 기법들 중 하나인 Weight pruning은 계산 그래프에서 edge를 줄여 계산량을 감소
- ADMM 기반 Weight pruning은 non-convex 최적화 식을 효과적으로 처리하여 빠른 시간 내에 성능 손실 없이 효과적인 sparsity 달성 가능
- 하지만 ADMM 기반의 방법들은 구조적으로 제거 비율을 설정하므로 현실적으로 Weight pruning이 필요한 큰 모델에 적용하기 힘듬
- 본 논문에서는 전역으로 제거 비율을 설정하여 ADMM 기반 Weight pruning을 수행하여 레이어 별 제거 비율을 자동으로 설정

## 사용 환경

CUDA Toolkit version은 10.1 이상을 권장

```
tensorflow-gpu==2.3.0rc0
opencv-python==4.1.1.26
numpy
lxml
tqdm
absl-py
matplotlib
easydict
pillow
```

## 초기 설정

#### 1) Gradual_increase_layer 및 LeNet-5
- 폴더 내에 train.py 실행

#### 2) Resnet-56
- 폴더 내에 train.py 실행


## 제안 모델 실행

각 모델은 훈련된 가중치를 기반으로 수행하며, 각 폴더 내에서 작동


#### 파라미터

```
k_step : ADMM step number
epochs : ADMM step(W update) training epoch
retraining_epochs : After ADMM step, retraining epoch
steps_per_epoch : Number of steps per epoch
rho : ADMM loss penalty parameter
p_lambda : Weight regulization parameter
all_percent : Removal percent
learning_rate
batch_size
```

#### 1) Gradual_increase_layer 및 LeNet-5

```
python admm_pruning.py
```

#### 2) Resnet-56

```
python admm_pruning_gpu.py
```



## 결과



## Contributiong / 기여자

* 양동욱(dongwook412@naver.com)
* 황보성훈(thehb01@gmail.com)
