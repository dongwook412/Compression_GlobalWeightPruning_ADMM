# 업데이트중

# ADMM 기반 전역 가중치 제거를 통한 딥러닝 모델의 압축

- 딥러닝 모델이 높은 성능을 달성하기 위해서는 모델의 가중치 수가 많아진다는 문제점 존재
- Model Compression은 메모리를 절약하고 모델의 저장 크기를 감소시키며 계산 요구사항을 줄일 수 있어 매우 유용
- Model Compression의 기법들 중 하나인 Weight pruning은 계산 그래프에서 edge를 줄여 계산량을 감소
- ADMM 기반 Weight pruning은 non-convex 최적화 식을 효과적으로 처리하여 빠른 시간 내에 성능 손실 없이 효과적인 sparsity 달성 가능
- 하지만 ADMM 기반의 방법들은 구조적으로 제거 비율을 설정하므로 현실적으로 Weight pruning이 필요한 큰 모델에 적용하기 힘듬
- 본 논문에서는 전역으로 제거 비율을 설정하여 ADMM 기반 Weight pruning을 수행하여 레이어 별 제거 비율을 자동으로 설정
- 점진적으로 레이어를 추가한 모델, LeNet-5와 같이 작은 모델뿐만 아니라 AlexNet, YOLOv4와 같은 비교적 큰 모델에 사용할 수 있음을 실험으로 보임

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

1. Gradual_increase_layer 및 LeNet-5
- 폴더 내에 train.py 실행

2. AlexNet
- [Alexnet.weights](https://www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy)에서 다운로드한 파일을 ./AlexNet/ 경로에 추가
- [ImageNet_ILSVRC2012](https://imagenet.stanford.edu/challenges/LSVRC/index.php)에서 ImageNet 데이터 다운로드
- 폴더 내에 preprocess_imagenet.py 실행하여 레이블 수정

3. YOLOv4
- 업데이트중

## 제안 모델 실행

각 모델은 훈련된 가중치를 기반으로 수행하며, 각 폴더 내에서 작동


### 파라미터

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

1. Gradual_increase_layer 및 LeNet-5

```
python admm_pruning.py
```

2. AlexNet

```
python admm_ADMM_step.py
python admm_Retraining_step.py
```

3. YOLOv4

```
예시
```

### 테스트는 이런 식으로 작성하시면 됩니다

```
예시
```

## Deployment / 배포

Add additional notes about how to deploy this on a live system / 라이브 시스템을 배포하는 방법

## Built With / 누구랑 만들었나요?

* [이름](링크) - 무엇 무엇을 했어요
* [Name](Link) - Create README.md

## Contributiong / 기여

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us. / [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) 를 읽고 이에 맞추어 pull request 를 해주세요.

## License / 라이센스

This project is licensed under the MIT License - see the [LICENSE.md](https://gist.github.com/PurpleBooth/LICENSE.md) file for details / 이 프로젝트는 MIT 라이센스로 라이센스가 부여되어 있습니다. 자세한 내용은 LICENSE.md 파일을 참고하세요.

## Acknowledgments / 감사의 말

* Hat tip to anyone whose code was used / 코드를 사용한 모든 사용자들에게 팁
* Inspiration / 영감
* etc / 기타
