# 업데이트중

# ADMM 기반 전역 가중치 제거를 통한 딥러닝 모델의 압축

- 딥러닝 모델이 높은 성능을 달성하기 위해서는 모델의 가중치 수가 많아진다는 문제점 존재
- Model Compression은 메모리를 절약하고 모델의 저장 크기를 감소시키며 계산 요구사항을 줄일 수 있어 매우 유용
- Model Compression의 기법들 중 하나인 Weight pruning은 계산 그래프에서 edge를 줄여 계산량을 감소
- ADMM 기반 Weight pruning은 non-convex 최적화 식을 효과적으로 처리하여 빠른 시간 내에 성능 손실 없이 효과적인 sparsity 달성 가능
- 하지만 ADMM 기반의 방법들은 구조적으로 제거 비율을 설정하므로 현실적으로 Weight pruning이 필요한 큰 모델에 적용하기 힘듬
- 본 논문에서는 전역으로 제거 비율을 설정하여 ADMM 기반 Weight pruning을 수행하여 레이어 별 제거 비율을 자동으로 설정
- 점진적으로 레이어를 추가한 모델, LeNet-5와 같이 작은 모델뿐만 아니라 AlexNet, YOLOv4와 같은 비교적 큰 모델에 사용할 수 있음을 실험으로 보임

## 시작하기

### 사용 환경

- CUDA Toolkit version은 10.1 이상을 권장

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

### 학습된 AlexNet weights 가져오기
- [Alexnet.weights](https://www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy)에서 다운로드한 파일을 ./AlexNet/ 경로에 추가
- 
## Running the tests / 테스트의 실행

어떻게 테스트가 이 시스템에서 돌아가는지에 대한 설명을 합니다

### 테스트는 이런 식으로 동작합니다

왜 이렇게 동작하는지, 설명합니다

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
