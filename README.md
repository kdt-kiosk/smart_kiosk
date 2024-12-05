# smart_kiosk
Title: 사용자 맞춤형 키오스크 
Subtitle: 누구나 쉽게 주문할 수 있는 시스템
팀원 : 박광준 , 이준성  
![image](https://github.com/user-attachments/assets/33a34af3-22a5-435e-b9eb-1d08d0dd4153)


## Backgroud
키오스크 설치 증가
고령층 장벽
외국인 장벽

배경설명 좀더자세


# Dataset
## 1. 감정분류 (한국인 데이터)
  (Source: ESTsoft 공통 + AIhub 한국인 얼굴합성을 위한 발화한 이미지)



| 버전 | 일자       | 변경내용          | 데이터 장수    |
| ---- | ---------- | ----------------- | -------------- |
| 1.0  | 2024-09-10 | 데이터 최종 개방 | 총 14,694장   |



## 항목별 데이터 분포

### 성별 분포

| 성별 | 건수 (개) | 비율   |
| ---- | --------- | ------ |
| 여자 | 8,085     | 55.0%  |
| 남자 | 6,609     | 45.0%  |

---

### 나이 분포

| 나이   | 건수 (개) | 비율   |
| ------ | --------- | ------ |
| 0-19   | 242       | 1.6%   |
| 20-39  | 11,517    | 78.4%  |
| 40 이상 | 2,935     | 20.0%  |

---

### 표정 분포

| 표정     | 건수 (개) | 비율   |
| -------- | --------- | ------ |
| Surprise | 2,099     | 14.3%  |
| Neutral  | 2,100     | 14.3%  |
| Fear     | 2,100     | 14.3%  |
| Disgust  | 2,100     | 14.3%  |
| Happy    | 2,095     | 14.3%  |
| Sad      | 2,100     | 14.3%  |
| Angry    | 2,100     | 14.3%  |


## 2. 성별/ 연령/ 인종 분류 (Asian + 외국인데이터) 
     (Source: ESTsoft 공통 + AIhub 한국인 얼굴합성을 위한 발화한 이미지 + UTK face ) 

| 버전 | 일자       | 변경내용          | 데이터 장수    |
| ---- | ---------- | ----------------- | -------------- |
| 1.0  | 2024-09-10 | 데이터 최종 개방 | 총 18,294장   |

### 성별 분포

| 성별 | 건수 (개) | 비율   |
| ---- | --------- | ------ |
| 여자 | 9,864     | 53.9%  |
| 남자 | 8,430     | 46.1%  |

---

### 나이 분포

| 나이   | 건수 (개) | 비율   |
| ------ | --------- | ------ |
| 0-19   | 983       | 5.4%   |
| 20-39  | 13,331    | 72.9%  |
| 40 이상 | 3,980     | 21.8%  |

---

### 인종 분포

| 인종         | 건수 (개) | 비율   |
| ------------ | --------- | ------ |
| 아시안       | 15,295    | 83.6%  |
| 그 외 인종   | 2,999     | 16.4%  |




- 연구 목적으로 활용되는 얼굴 데이터베이스는 대부분 서양인을 대상으로 구축 
- 아시아인 데이터를 추가함


------------------
# 파이프라인
![맞춤형키오스크 개념도_develop drawio](https://github.com/user-attachments/assets/1a2283a8-66bc-4395-b0a8-65fd59fa30ab)





----
# 사용한 모델
  ## Hyperparameter Sweep Results

   이 문서는 모델 학습을 위해 실행된 하이퍼파라미터 스윕의 결과를 요약한 내용입니다. 아래는 각 스윕과 그 결과에 대한 세부 정보입니다.

  ### Sweep Overview

  모델 학습을 위한 최적의 하이퍼파라미터를 찾기 위해 다양한 `batch_size`, `dropout_rate`, `epochs`, `l2_reg`, `learning_rate` 값을 실험하였습니다.

https://huggingface.co/kdtFinalProject 참조
-----

** val_auc 기준 sweep 상위 10개 지표 정리
<details>
  <summary><h1> 감정분류</h1></summary>
  <!-- 내용 -->
</details>


<details>
  <summary><h1> 인종분류</h1></summary>
  
  # VGG16 모델 (사전학습된 모델을 fine-tuning 및 tranfer Learning)
   ## Summary of Results

  | Sweep Name         | Epochs | Best Epoch | Batch Size | Dropout Rate | L2 Regularization | Learning Rate | Test Accuracy | Train Accuracy | Validation Accuracy |
  |--------------------|--------|------------|------------|--------------|--------------------|---------------|---------------|----------------|----------------------|
  | giddy-sweep-11     | 15     | 15         | 32         | 0.2          | 0.0001             | 0.0001        | 0.9777        | 0.9963         | 0.9781               |
  | vivid-sweep-12     | 15     | 15         | 64         | 0.1          | 0.000001           | 0.0001        | 0.9781        | 1.0000         | 0.9777               |
  | sage-sweep-16      | 15     | 15         | 128        | 0.1          | 0.000001           | 0.0001        | 0.9792        | 1.0000         | 0.9769               |
  | rare-sweep-7       | 15     | 15         | 128        |              | 0.00001            | 0.0004        | 0.9796        | 1.0000         | 0.9762               |
  | honest-sweep-9     | 10     | 10         | 64         | 0.3          | 0.000001           | 0.0004        | 0.9796        | 0.9972         | 0.9762               |
  | volcanic-sweep-20  | 15     | 4          | 128        | 0.2          | 0.000001           | 0.0001        | 0.9792        | 0.9979         | 0.9758               |
  | lively-sweep-18    | 15     | 15         | 128        |              | 0.000001           | 0.0001        | 0.9800        | 1.0000         | 0.9754               |
  | soft-sweep-15      | 15     | 11         | 64         | 0.2          | 0.0001             | 0.0001        | 0.9800        | 0.9987         | 0.9742               |
  | smooth-sweep-8     | 10     | 10         | 64         | 0.1          | 0.0001             | 0.0004        | 0.9792        | 0.9998         | 0.9738               |
  | daily-sweep-13     | 15     | 15         | 128        | 0.3          | 0.000001           | 0.0004        | 0.9785        | 0.9994         | 0.9738               |

  ## Key Insights

  1. **Best Performance**: `lively-sweep-18`은 **98.00%**의 가장 높은 테스트 정확도를 달성하였습니다.
     - Batch Size: 128
     - Dropout Rate: 
     - Epochs: 15
     - L2 Regularization: 0.000001
     - Learning Rate: 0.0001

  2. **Optimal Trade-off**: `giddy-sweep-11`와 `sage-sweep-16`은 효율적인 실행 시간으로 약 97%의 성능을 제공했습니다.

  3. **Validation Observations**: 검증 정확도는 대부분 97.15%에서 97.81% 사이로 안정적인 일반화 성능을 보였습니다.
  ---
  # MobilenetV4 (모델구조만 가져와 fine-tuning 및 tranfer Learning)  
  ## Summary of Results

  | Sweep Name         | Epochs | Best Epoch | Batch Size | Dropout Rate | L2 Regularization | Learning Rate | Test Accuracy | Train Accuracy | Validation Accuracy |
  |--------------------|--------|------------|------------|--------------|--------------------|---------------|---------------|----------------|----------------------|
  | vague-sweep-16     | 10     | 5          | 128        | 0.2          | 0.000001           | 0.01          | 0.9665        | 0.9672         | 0.9700               |
  | winter-sweep-2     | 15     | 4          | 32         |              | 0.0001             | 0.0004        | 0.9692        | 0.9682         | 0.9673               |
  | serene-sweep-6     | 10     | 4          | 16         | 0.2          | 0.00001            | 0.0004        | 0.9665        | 0.9655         | 0.9673               |
  | driven-sweep-15    | 15     | 5          | 32         | 0.3          | 0.0001             | 0.0004        | 0.9719        | 0.9687         | 0.9669               |
  | generous-sweep-12  | 15     | 4          | 64         | 0.2          | 0.000001           | 0.01          | 0.9669        | 0.9596         | 0.9662               |
  | fluent-sweep-13    | 15     | 4          | 128        |              | 0.000001           | 0.001         | 0.9650        | 0.9705         | 0.9642               |
  | ethereal-sweep-3   | 10     | 5          | 32         | 0.1          | 0.000001           | 0.01          | 0.9623        | 0.9602         | 0.9635               |
  | visionary-sweep-19 | 15     | 5          | 32         | 0.3          | 0.0001             | 0.01          | 0.9542        | 0.9546         | 0.9623               |
  | olive-sweep-20     | 10     | 4          | 32         | 0.3          | 0.000001           | 0.0004        | 0.9685        | 0.9672         | 0.9619               |
  | whole-sweep-5      | 15     | 5          | 128        | 0.2          | 0.0001             | 0.01          | 0.9619        | 0.9591         | 0.9615               |

  ## Key Insights

  1. **Best Performance**:  `driven-sweep-15`은 **97.19%**의 가장 높은 테스트 정확도를 달성하였습니다.
     - Batch Size: 32
     - Dropout Rate: 0.3
     - Epochs: 15
     - L2 Regularization: 0.0001
     - Learning Rate: 0.0004

  2. **Optimal Trade-off**:  `winter-sweep-2`와 `serene-sweep-6`은 효율적인 실행 시간으로 약 96%의 성능을 제공했습니다.

  3. **Validation Observations**: 검증 정확도는 대부분 96.15%에서 97.00% 사이로 안정적인 일반화 성능을 보였습니다.
</details>

<details>
  <summary><h1>성별분류</h1></summary>


  # ResNet18 (사전학습된 모델을 fine-tuning 및 tranfer Learning)
  ## Summary of Results

 | Sweep Name         | Epochs | Best Epoch | Batch Size | Dropout Rate | L2 Regularization | Learning Rate | Test Accuracy | Train Accuracy | Validation Accuracy |
  |--------------------|--------|------------|------------|--------------|--------------------|---------------|---------------|----------------|----------------------|
  | easy-sweep-11      | 10     | 7          | 32         | 0.3          | 0.000001           | 0.0001        | 0.9727        | 0.9960         | 0.9681               |
  | sweepy-sweep-17    | 10     | 4          | 16         | 0.3          | 0.000001           | 0.0001        | 0.9712        | 0.9818         | 0.9638               |
  | woven-sweep-15     | 10     | 5          | 16         | 0.3          | 0.000001           | 0.0001        | 0.9612        | 0.9855         | 0.9635               |
  | cool-sweep-9       | 15     | 11         | 32         | 0.3          | 0.000001           | 0.0001        | 0.9619        | 0.9946         | 0.9635               |
  | golden-sweep-6     | 15     | 4          | 64         | 0.3          | 0.00001            | 0.0001        | 0.9692        | 0.9885         | 0.9615               |
  | snowy-sweep-10     | 15     | 6          | 32         | 0.3          | 0.000001           | 0.0001        | 0.9635        | 0.9914         | 0.9612               |
  | swift-sweep-18     | 10     | 4          | 16         | 0.3          | 0.000001           | 0.0001        | 0.9635        | 0.9800         | 0.9612               |
  | charmed-sweep-14   | 15     | 8          | 32         | 0.3          | 0.000001           | 0.0001        | 0.9635        | 0.9949         | 0.9600               |
  | avid-sweep-20      | 10     | 8          | 16         | 0.3          | 0.00001            | 0.0001        | 0.9650        | 0.9895         | 0.9596               |
  | generous-sweep-12  | 15     | 4          | 32         | 0.3          | 0.000001           | 0.0001        | 0.9642        | 0.9874         | 0.9592               |

  ## Key Insights

  1. **Best Performance**: `easy-sweep-11`이 **97.27%**의 가장 높은 테스트 정확도를 달성하였습니다
     - Batch Size: 32
     - Dropout Rate: 0.3
     - Epochs: 10
     - L2 Regularization: 0.000001
     - Learning Rate: 0.0001

  2. **Optimal Trade-off**: `sweepy-sweep-17`과 `swift-sweep-18`은 약 96%의 테스트 정확도를 제공하면서도 실행 시간이 20분 미만으로 짧았습니다.

  3. **Validation Observations**: 대부분의 sweep에서 검증 정확도가 95.92%에서 96.81%로 안정적으로 유지되며, 일반화가 잘 이루어졌음을 보여줍니다.
----
# 다른모델


</details>


<details>
  <summary><h1> 연령분류</h1></summary>
  # ResNet18 (사전학습된 모델을 fine-tuning 및 tranfer Learning)

  ## Summary of Results

  | Sweep Name         | Epochs | Best Epoch | Batch Size | Dropout Rate | L2 Regularization | Learning Rate | Test Accuracy | Train Accuracy | Validation Accuracy |
  |--------------------|--------|------------|------------|--------------|--------------------|---------------|---------------|----------------|----------------------|
  | splendid-sweep-9   | 10     | 7          | 128        | 0.3          | 0.000001           | 0.0001        | 0.8085        | 0.9875         | 0.8162               |
  | deft-sweep-16      | 10     | 10         | 128        |              | 0.0001             | 0.0001        | 0.8077        | 0.9906         | 0.8158               |
  | charmed-sweep-5    | 15     | 12         | 128        | 0.2          | 0.000001           | 0.0001        | 0.8050        | 0.9916         | 0.8131               |
  | autumn-sweep-13    | 10     | 9          | 128        |              | 0.000001           | 0.0001        | 0.8054        | 0.9909         | 0.8123               |
  | glorious-sweep-10  | 15     | 5          | 128        | 0.3          | 0.000001           | 0.0001        | 0.8173        | 0.9828         | 0.8115               |
  | stellar-sweep-10   | 10     | 10         | 128        |              | 0.00001            | 0.0001        | 0.8104        | 0.9902         | 0.8112               |
  | devoted-sweep-15   | 10     | 10         | 128        |              | 0.000001           | 0.0001        | 0.8100        | 0.9885         | 0.8112               |
  | decent-sweep-19    | 10     | 7          | 128        |              | 0.0001             | 0.0001        | 0.8023        | 0.9869         | 0.8104               |
  | golden-sweep-20    | 10     | 10         | 128        |              | 0.00001            | 0.0001        | 0.8050        | 0.9898         | 0.8096               |
  | giddy-sweep-19     | 10     | 4          | 128        | 0.2          | 0.000001           | 0.0001        | 0.7946        | 0.9767         | 0.8081               |

  ## Key Insights

  1. **Best Performance**:  `glorious-sweep-10`은 **81.73%**의 가장 높은 테스트 정확도를 달성하였습니다
     - Batch Size: 128
     - Dropout Rate: 0.3
     - Epochs: 15
     - L2 Regularization: 0.000001
     - Learning Rate: 0.0001

  2. **Optimal Trade-off**:  `stellar-sweep-10`과 `deft-sweep-16`은 효율적인 실행 시간으로 약 81%의 성능을 제공했습니다.

  3. **Validation Observations**: 검증 정확도는 대부분 80.81%에서 81.62% 사이로 안정적인 일반화 성능을 보였습니다.

</details>

만약 토글이 안된다면 따로 페이지 생성해서 작성
----

ex) 분류모델별 성능비교
https://cdn.discordapp.com/attachments/1229239889799807042/1313442062179504200/image.png?ex=675025be&is=674ed43e&hm=d8e88e70b6580a5ff28bbe358bb6eef2e32a9aa447ef3efcab8585059d6f32dd&

### 인종분류 Best-Performing Model Comparison

| Model          | Test Accuracy | Test F1 Score | Test Precision | Test Recall | Train Accuracy | Validation Accuracy | Validation Loss |
|----------------|---------------|---------------|----------------|-------------|----------------|---------------------|-----------------|
| **MobileNetV4**| **0.9719**    | **0.9724**    | **0.9737**     | **0.9719**  | 0.9686         | 0.9683              | 0.0787          |
| **VGG16**      | **0.9807**    | **0.9809**    | **0.9811**     | **0.9807**  | 0.9997         | 0.9727              | 0.1188          |

다음과 같은 이유로 **MobilenetV4**를 인종분류모델로써 선택했습니다
- mobilnetV4가 VGG16에 비해 아키텍처가 더 작고, 메모리 요구 사항이 낮으며, 추론 시간이 더 빠릅니다.
- 키오스크는 사용자에게 즉각적인 피드백을 제공하기 위해 실시간 처리가 필요한 경우가 많습니다.
- MobileNetV4는 스윕에서 검증 손실이 더 낮았으며, 이는 다양한 사용자가 시스템과 상호 작용하는 키오스크에서 중요한, 보이지 않는 데이터에 대해 잘 일반화되었음을 나타냅니다.  




~~ 선택된 최종모델
## 최종사용모델
감정분류 [DDAMNet](https://github.com/SainingZhang/DDAMFN) 

<details>
  <summary><h3>인종분류 MobilenetV4</h3></summary>
  
  [MobilenetV4 info](https://huggingface.co/blog/rwightman/mobilenetv4)   
  [MobileNetV4 Github](https://github.com/tensorflow/models/blob/master/official/vision/modeling/backbones/mobilenet.py)  
- UIB(Universal Inverted Bottleneck): 이 유연한 구조는 IB(Inverted Bottleneck), ConvNext, FFN(Feed Forward Network) 및 새로운 ExtraDW(ExtraDW) 변형을 포함한 다양한 아키텍처 요소를 통합합니다. 이러한 요소는 집합적으로 모델의 성능과 적응성을 향상시킵니다.
  
  ![image](https://github.com/user-attachments/assets/ae8d7751-c57b-491e-b2a0-90ebf3736e56)

- 모바일 MQA: 모바일 액셀러레이터용으로 설계된 특수 어텐션 블록인 Mobile MQA는 기존 어텐션 메커니즘에 비해 39% 상당한 속도 향상을 달성하여 모바일 하드웨어의 추론 효율성을 높입니다.
- 최적화된 신경 아키텍처 검색(NAS): 이 향상된 NAS 방법론은 아키텍처 검색 효율성을 향상시켜 CPU, DSP, GPU 및 Apple Neural Engine 및 Google Pixel EdgeTPU와 같은 특수 가속기를 포함한 다양한 모바일 플랫폼에서 대부분 Pareto 최적인 모델을 만듭니다.
</details>




<details>
  <summary><h3> 성별분류,연령분류 ResNet18</h3></summary>
  
 [ResNet18 docs](https://huggingface.co/docs/transformers/model_doc/resnet)    
 [ResNet18 github](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)


![image](https://github.com/user-attachments/assets/7f4bd0e9-0b78-401f-9d41-cf6978933edd)

  ResNet18은 18개의 층으로 이루어진 ResNet을 의미하는데 244*244*3의 image를 input으로 받는다. 그리고 4개의 Conv block과 Adaptive Average Pooling을 시행한 후 Fully Connected layer(FC layer)를 통과시켜 이미지를 분류한다. 각각의 Conv block은 두 개의 3*3 Conv 레이어로 구성되어 있다.
  
- **연결 건너뛰기(잔여 연결)**:
  ResNet18은 하나 이상의 레이어를 우회하는 잔여 연결을 도입하여 기울기가 이전 레이어로 직접 흐를 수 있도록 합니다. 이는 Vanishing Gradient 문제를 효과적으로 해결하고 더 깊은 네트워크의 훈련을 지원합니다.
- **아키텍처 개요:**
  1. **첫 번째 컨볼루션 레이어:**
입력에서 신경망이 특징을 추출하도록 3×3크기의 필터로 입력 데이터를 합성곱  
  2. **활성화 함수 (ReLU):**  
비선형성을 추가하여 네트워크가 복잡한 패턴을 학습할 수 있도록 컨볼루션 결과에 ReLU 활성화 함수를 적용합니다.  
  3. **두 번째 컨볼루션 레이어:**  
3×3 크기의 필터로 특징을 추출, 첫 번째 레이어의 출력과 동일한 수의 필터를 사용하여 차원을 일치시킵니다.  
  4. **잔차 연결 (Residual Connection):**  
두 번째 컨볼루션 레이어의 출력에 첫 번째 레이어의 입력을 직접 더함. 이는 잔차를 학습하도록 하며, 기울기 소실 문제를 완화시켜 더 깊은 네트워크를 효과적으로 훈련시킴.  
  5. **활성화 함수 (ReLU):**  
잔차 연결 후에도 ReLU 활성화 함수를 적용하여 비선형성을 유지   
</details>




얼굴인식 [yolov11n-face](https://github.com/akanametov/yolo-face)  
시선추정 [roboflow](https://blog.roboflow.com/gaze-direction-position/)    


-- 각 최종모델에 대한 상세 설명 추가 (원리  ,구조 )
---
# 시선추적 관련 calibration 생성
https://github.com/kdt-kiosk/kiosk_gaze



--- 
# UI 설명 간단히
recommend 중심으로 해야함
언어선택도 포함
senior UI에 대한 설

# 시연 동영상

** 40세 이상인 아닌 사람이 일반 UI로 이용하는 모습
분류모델로 이용해서 추천메뉴를 사용자의 상태에 맞게 설정가능 




https://github.com/user-attachments/assets/f4172814-cdc7-4af5-85de-da0e1a692c96  


** 40세 이상인  사람이  전용UI로 시선추적으로 이용하는 모습


https://github.com/user-attachments/assets/54ce00b6-655c-4a77-b058-c0359bc9db30


# 결론
AI 모델을 이용해서 사용자 맞춤 서비스를 이용할수있다
취약층 계층에게 시선으로만 주문이 가능하게 만들어 접근성을 높인다. <- 이런식으로 작성 Backgroud 참고


