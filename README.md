# smart_kiosk
Title: 사용자 맞춤형 키오스크 
Subtitle: 누구나 쉽게 주문할 수 있는 시스템
팀원 : 박광준 , 이준성  
![image](https://github.com/user-attachments/assets/33a34af3-22a5-435e-b9eb-1d08d0dd4153)





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
  <summary><h2> 감정분류</h2></summary>
  
  ## 1. PAtt-Lite 모델 (사전학습된 모델을 fine-tuning 및 tranfer Learning)
  
  ### Summary of Results

  | Sweep Name         | Epochs | Best Epoch | Batch Size | Dropout Rate | L2 Regularization | Learning Rate | Test Accuracy | Train Accuracy | Validation Accuracy |
  |--------------------|--------|------------|------------|--------------|-------------------|---------------|---------------|----------------|---------------------|
  | rich-sweep-5       | 20     | 7          | 8          | 0.1          | 0.00001           | 0.01          | 0.8694        | 0.8767         | 0.8833              |
  | cool-sweep-8       | 30     | 7          | 16         | 0.45         | 0.0001            | 0.001         | 0.8832        | 0.9122         | 0.8805              |
  | lemon-sweep-23     | 20     | 12         | 8          | 0.3          | 0.00001           | 0.05          | 0.8827        | 0.8695         | 0.8786              |
  | logical-sweep-9    | 30     | 12         | 64         | 0.45         | 0                 | 0.0001        | 0.8718        | 0.9088         | 0.8733              |
  | gallant-sweep-26   | 20     | 5          | 8          | 0.3          | 0.00001           | 0.001         | 0.8675        | 0.8880         | 0.8724              |
  | soft-sweep-2       | 20     | 12         | 16         | 0.3          | 0.001             | 0.01          | 0.8675        | 0.9025         | 0.8686              |
  | chocolate-sweep-14 | 30     | 7          | 16         | 0.45         | 0                 | 0.001         | 0.8894        | 0.9124         | 0.8667              |
  | young-sweep-11     | 20     | 8          | 8          | 0.3          | 0.001             | 0.01          | 0.8675        | 0.8579         | 0.8657              |
  | eager-sweep-10     | 20     | 12         | 16         | 0.5          | 0                 | 0.0001        | 0.8913        | 0.9225         | 0.8648              |
  | olive-sweep-28     | 20     | 8          | 8          | 0.4          | 0.1               | 0.001         | 0.8732        | 0.8801         | 0.8643              |

  ## Key Insights

1. **최고 테스트 정확도:**
   - `eager-sweep-10`이 **89.13%**의 가장 높은 테스트 정확도를 달성했습니다.
     <ul>
       <li>Batch Size: 16</li>
       <li>Dropout Rate: 0.5</li>
       <li>Learning Rate: 0.0001</li>
     </ul>

2. **드롭아웃 비율의 영향:**
   - 높은 드롭아웃 비율(예: `0.45`와 `0.5`)은 대체로 테스트 정확도를 향상시켰습니다.
   - 예외적으로 `gallant-sweep-26`은 드롭아웃 비율이 `0.3`임에도 낮은 테스트 정확도를 보였습니다.

3. **학습률(Learning Rate)의 영향:**
   - 낮은 학습률(예: `0.0001`)은 일반적으로 더 나은 결과를 가져왔으며, 특히 `eager-sweep-10`에서 뚜렷한 성능 향상을 보였습니다.

4. **배치 크기(Batch Size)의 관찰:**
   - 큰 배치 크기(예: `logical-sweep-9`의 `64`)는 반드시 더 나은 테스트 정확도를 보장하지 않았습니다. 이는 과도한 일반화(overgeneralization) 때문일 가능성이 높습니다.

5. **훈련 정확도와 테스트 정확도의 일관성:**
   - `eager-sweep-10`과 `chocolate-sweep-14`는 훈련 정확도(약 92%)와 테스트 정확도(약 89%) 사이에서 높은 일관성을 유지했습니다.

  ---

  ## 2. DDAMFN++ 모델 (사전학습된 모델을 fine-tuning 및 tranfer Learning)

  ### Summary of Results
  
  | Sweep Name         | Epochs | Batch Size | Dropout Rate | L2 Regularization | Learning Rate | Test Accuracy | Train Accuracy | Validation Accuracy |
  |--------------------|--------|------------|--------------|-------------------|---------------|---------------|----------------|---------------------|
  | dauntless-sweep-2  | 30     | 32         | 0.1          | 0.0001            | 0.0001        | 0.9252        | 0.8965         | 0.9310              |
  | different-sweep-6  | 20     | 32         | 0.1          | 0.0001            | 0.0001        | 0.9357        | 0.9009         | 0.9300              |
  | eager-sweep-3      | 30     | 32         | 0.1          | 0.0001            | 0.0001        | 0.9329        | 0.8968         | 0.9262              |
  | cool-sweep-15      | 30     | 64         | 0.5          | 0.001             | 0.0001        | 0.9205        | 0.7808         | 0.9238              |
  | pious-sweep-1      | 30     | 32         | 0.1          | 0.0001            | 0.0001        | 0.9233        | 0.8899         | 0.9238              |
  | dulcet-sweep-7     | 30     | 64         | 0.3          | 0.001             | 0.0001        | 0.9257        | 0.8529         | 0.9171              |
  | mild-sweep-3       | 30     | 64         | 0.1          | 0.001             | 0.00001       | 0.9267        | 0.8711         | 0.9171              |
  | lyric-sweep-11     | 30     | 128        | 0.5          | 0.001             | 0.0001        | 0.9290        | 0.7931         | 0.9167              |
  | classic-sweep-5    | 30     | 128        | 0.5          | 0.001             | 0.0001        | 0.9295        | 0.7848         | 0.9162              |
  | silver-sweep-12    | 20     | 128        | 0.3          | 0.001             | 0.0001        | 0.9281        | 0.8566         | 0.9152              |
  
  
  
  ### Key Insights
  
  1. **최고 테스트 정확도:**
     - `different-sweep-6`이 **93.57%**의 가장 높은 테스트 정확도를 달성했습니다.
       <ul>
         <li>Batch Size: 32</li>
         <li>Dropout Rate: 0.1</li>
         <li>Learning Rate: 0.0001</li>
       </ul>
  
  2. **드롭아웃 비율의 영향:**
     - 낮은 드롭아웃 비율(예: `0.1`)은 대체로 더 높은 테스트 정확도를 기록했습니다.
     - 그러나 `cool-sweep-15`와 같은 높은 드롭아웃 비율(0.5)은 낮은 훈련 정확도와 일관되지 않은 성능을 보였습니다.
  
  3. **학습률(Learning Rate)의 중요성:**
     - 모든 실험에서 **0.0001**의 학습률이 사용되었으며, 이는 일관된 성능을 유지하는 데 긍정적인 영향을 미쳤습니다.
  
  4. **배치 크기(Batch Size)의 관찰:**
     - 작은 배치 크기(32)는 높은 정확도를 달성하는 데 유리했으며, 특히 `different-sweep-6`과 같은 모델에서 두드러졌습니다.
     - 큰 배치 크기(128)는 훈련 정확도와 테스트 정확도의 일관성이 떨어지는 경향이 있었습니다.
  
  5. **훈련과 테스트 정확도의 일치:**
     - `eager-sweep-3`와 `dauntless-sweep-2`는 훈련 정확도(약 89%)와 테스트 정확도(약 93%) 사이에서 높은 일관성을 유지했습니다.
  
  ---

</details>


<details>
  <summary><h2> 인종분류</h2></summary>
  
  ## 1.VGG16 모델 (사전학습된 모델을 fine-tuning 및 tranfer Learning)
   ### Summary of Results

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
  ## 2.MobilenetV4 (모델구조만 가져와 fine-tuning 및 tranfer Learning)  
  ### Summary of Results

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
  <summary><h2>성별분류</h2></summary>


  ## 1.ResNet18 (사전학습된 모델을 fine-tuning 및 tranfer Learning)
  ### Summary of Results

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



 ## 2. Mini-Xception 모델 (사전학습된 모델을 fine-tuning 및 tranfer Learning)
  ### Summary of Results
  
  | Sweep Name         | Epochs | Best Epoch | Batch Size | Dropout Rate | L2 Regularization | Learning Rate | Test Accuracy | Train Accuracy | Validation Accuracy |
  |--------------------|--------|------------|------------|--------------|-------------------|---------------|---------------|----------------|---------------------|
  | sweepy-sweep-17    | 20     | 6          | 32         | 0.55         | 0.1               | 0.000001      | 0.8050        | 0.7441         | 0.8062              |
  | eternal-sweep-21   | 30     | 4          | 64         | 0.2          | 0.00001           | 0.000005      | 0.8104        | 0.7892         | 0.8058              |
  | winter-sweep-18    | 20     | 5          | 16         | 0.15         | 0.0001            | 0.0001        | 0.8119        | 0.7921         | 0.8058              |
  | grateful-sweep-20  | 20     | 4          | 16         | 0.5          | 0.1               | 0.000001      | 0.8096        | 0.7517         | 0.8054              |
  | volcanic-sweep-27  | 30     | 3          | 16         | 0.2          | 0                 | 0.0001        | 0.8142        | 0.7887         | 0.8054              |
  | light-sweep-19     | 20     | 4          | 64         | 0.3          | 0.0001            | 0.000005      | 0.8108        | 0.7785         | 0.8050              |
  | spring-sweep-29    | 30     | 3          | 32         | 0.25         | 0.1               | 0.000001      | 0.8127        | 0.7848         | 0.8050              |
  | hopeful-sweep-30   | 30     | 3          | 32         | 0.5          | 0.01              | 0.00001       | 0.8115        | 0.7571         | 0.8031              |
  | fearless-sweep-25  | 30     | 3          | 32         | 0            | 0.0001            | 0.0005        | 0.8123        | 0.8001         | 0.8031              |
  | sleek-sweep-16     | 30     | 8          | 64         | 0            | 0                 | 0.001         | 0.8054        | 0.7997         | 0.8027              |
  
  
  ---
  
  ### Key Insights
  
  1. **최고 테스트 정확도:**
     - `volcanic-sweep-27`이 **81.42%**의 가장 높은 테스트 정확도를 기록했습니다.
       <ul>
         <li>Batch Size: 16</li>
         <li>Dropout Rate: 0.2</li>
         <li>Learning Rate: 0.0001</li>
       </ul>
  
  2. **드롭아웃 비율의 영향:**
     - 높은 드롭아웃 비율(예: `0.5`)은 대체로 낮은 훈련 정확도와 일관되지 않은 테스트 정확도를 보였습니다.
     - 낮은 드롭아웃 비율(예: `0.2` 및 `0.25`)은 더 높은 테스트 정확도를 보장하는 경향이 있었습니다.
  
  3. **학습률(Learning Rate)의 영향:**
     - 낮은 학습률(예: `0.0001`)은 모델 안정성과 테스트 정확도를 높이는 데 효과적이었습니다.
  
  4. **배치 크기(Batch Size)의 관찰:**
     - 중간 배치 크기(32)가 대부분의 실험에서 높은 테스트 정확도를 기록했습니다.
     - 큰 배치 크기(64)는 테스트 정확도에 부정적인 영향을 미치는 경향이 있었습니다.
  
  5. **훈련과 테스트 정확도의 일치:**
     - `volcanic-sweep-27`과 `fearless-sweep-25`는 훈련 정확도(약 80%)와 테스트 정확도(약 81%) 사이에서 높은 일관성을 유지했습니다.
  


---



</details>


<details>
  <summary><h2> 연령분류</h2></summary>
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


----

## 모델별 성능 비교
### 감정 분류
| 모델 이름         | 정확도(%) | 정밀도(%) | 재현율(%) | F1 스코어(%) |
|------------------|----------|----------|----------|-------------|
| **DDAMFN++**       | 93.57     | 93.33     | 93.83     | 93.58        |
| **PAtt-Lite**       | 89.13     | 89.40     | 89.29     | 89.34        |
| **Deepface**       | 29.81     | 29.98     | 29.81     | 26.89        |

### 인종분류 

| Model          | Test Accuracy | Test F1 Score | Test Precision | Test Recall | 
|----------------|---------------|---------------|----------------|-------------|
| **MobileNetV4**| **0.9719**    | **0.9724**    | **0.9737**     | **0.9719**  | 
| **VGG16**      | **0.9807**    | **0.9809**    | **0.9811**     | **0.9807**  | 
| **Deepface**       | 77.31     | 40.46     | 88.70     | 55.57        |

다음과 같은 이유로 **MobilenetV4**를 인종분류모델로써 선택했습니다
- mobilnetV4가 VGG16에 비해 아키텍처가 더 작고, 메모리 요구 사항이 낮으며, 추론 시간이 더 빠릅니다.
- 키오스크는 사용자에게 즉각적인 피드백을 제공하기 위해 실시간 처리가 필요한 경우가 많습니다.
- MobileNetV4는 스윕에서 검증 손실이 더 낮았으며, 이는 다양한 사용자가 시스템과 상호 작용하는 키오스크에서 중요한, 보이지 않는 데이터에 대해 잘 일반화되었음을 나타냅니다.  

### 성별 분류
| Model          | Test Accuracy | Test F1 Score | Test Precision | Test Recall | 
|----------------|---------------|---------------|----------------|-------------|
| **Mini-Xception**       | 81.42     | 84.06     | 81.14     | 82.57        |
| **Deepface**       | 71.81     | 88.95     | 54.82     | 67.84        |

### 연령 분류
| Model          | Test Accuracy | Test F1 Score | Test Precision | Test Recall |
|----------------|---------------|---------------|----------------|-------------|
| **ResNet18**| **0.8173**    | **0.7970**    | **0.8135**     | **0.8173**  | 
| **모델 B**       | 12.3     | 12.3     | 12.3     | 12.3        |
| **Deepface**       | 75.77     | 78.75     | 40.41     | 41.48        |



## 최종사용모델
<details>
  <summary><h3>1. 감정분류 - DDAMNet</h3></summary>
  
  [DDAMNet GitHub](https://github.com/SainingZhang/DDAMFN)<br>
  [DDAMNet Paper](https://www.mdpi.com/2079-9292/12/17/3595)
  #### 구조 및 원리
  ![image](https://github.com/user-attachments/assets/b4ec1c20-1452-4018-a310-81e92cfd9a08)
  ##### 개요
  DDAMFN(Dual-Direction Attention Mixed Feature Network)은 얼굴 표정 인식(Facial Expression Recognition, FER)을 위해 설계된 경량화되고 강력한 딥러닝 모델입니다. 이 네트워크는 입력 이미지로부터 특징을 추출하고, 중요한 부분에 집중하여 얼굴 표정을 정확하게 예측합니다. DDAMFN은 크게 두 가지 주요 구성 요소로 나뉩니다:
  1. **MFN (Mixed Feature Network):** 얼굴 이미지로부터 기본적인 특징을 추출.
  2. **DDAN (Dual-Direction Attention Network):** 추출된 특징을 분석하고, 얼굴에서 중요한 부분에 주의를 집중.
  
  ##### 모델 구조
  
  ###### 1. **Feature Extraction (특징 추출)**
  - 입력 이미지 크기: \(112 \times 112\)
  - **MFN (Mixed Feature Network):**
    - 다양한 크기의 합성곱 커널(MixConv)을 사용하여 얼굴 이미지에서 세부적이고 전반적인 특징을 추출합니다.
    - Residual Bottleneck 구조를 통해 정보 손실을 방지하고, 효율적인 학습이 가능하도록 설계되었습니다.
    - Stride-1과 Stride-2 블록을 활용하여 세부 정보와 주요 특징을 모두 보존합니다.
    - **Coordinate Attention**을 적용하여 얼굴 이미지의 중요한 위치(눈, 코, 입 등)를 강조합니다.
    - 최종적으로 \(7 \times 7 \times 512\) 크기의 특징 맵을 생성합니다.
  
  ###### 2. **Attention Module (주의 모듈)**
  - **Dual Direction Attention (DDA):**
    - 두 가지 방향으로 주의를 분석:
      - **수평 방향(X Linear GDConv):** 얼굴의 좌우 영역을 분석.
      - **수직 방향(Y Linear GDConv):** 얼굴의 위아래 영역을 분석.
    - 두 방향의 결과를 결합(Concatenate & Conv2D)하여 종합적인 주의 맵을 생성합니다.
  - **주의 맵 생성:** 중요한 부분을 강조하고, 덜 중요한 영역의 영향을 줄입니다.
  - **MAX 연산:** 여러 주의 맵 중에서 가장 유용한 맵을 선택합니다.
  - **곱셈 연산:** 주의 맵과 기존 특징 맵을 결합하여, 모델이 얼굴의 중요한 영역에 집중할 수 있도록 합니다.
  
  ###### 3. **Feature Transformation (특징 변환)**
  - **Global Depthwise Convolution (GDConv):**
    - 주의 맵이 결합된 특징 맵을 압축하여 \(1 \times 1 \times 512\) 크기로 변환합니다.
  - **Reshape:** 특징 맵을 256차원의 벡터로 변환하여 최종 예측에 사용합니다.
  
  ###### 4. **Fully Connected Layer (완전 연결 계층)**
  - 변환된 256차원 벡터는 완전 연결 계층(FC Layer)을 통과하여, 입력 얼굴 이미지의 감정을 예측합니다.
  - **손실 함수(Loss Function):**
    - **Cross-Entropy Loss (\(L_{cls}\)):** 모델의 감정 예측 성능을 최적화합니다.
    - **Attention Loss (\(L_{att}\)):** 주의 모듈이 서로 다른 얼굴 영역에 집중하도록 유도합니다.
  
  ##### 동작 원리 요약
  1. **MFN (특징 추출):** 얼굴 이미지로부터 다양한 크기의 커널을 활용해 세부적이고 전반적인 특징을 추출.
  2. **DDAN (주의 모듈):** 두 방향(수평, 수직)으로 얼굴의 중요한 영역을 분석하고, 주의 맵을 생성.
  3. **Feature Transformation:** 추출된 특징을 압축하고 변환하여 예측 가능한 벡터로 변환.
  4. **FC Layer:** 최종 예측을 통해 얼굴 감정을 분류.
  
  ##### 모델의 강점
  1. **경량화:** 계산량을 줄이면서도 높은 정확도를 유지.
  2. **효율적인 특징 추출:** 다양한 크기의 커널과 Residual Bottleneck 구조를 활용.
  3. **주의 메커니즘:** 얼굴의 중요한 부분에 집중하여 예측 성능 향상.
  4. **적용 가능성:** 다양한 감정 인식 및 얼굴 기반 응용 프로그램에 활용 가능.
</details>


<details>
  <summary><h3>2.인종분류 MobilenetV4</h3></summary>
  
[MobileNetV4 논문](https://arxiv.org/abs/2404.10518) | [MobileNetV4 GitHub](https://github.com/tensorflow/models/blob/master/official/vision/modeling/backbones/mobilenet.py)

#### **구조 및 원리**
**MobileNetV4**는 경량화된 효율적인 딥러닝 모델로, 인종 분류와 같은 작업을 위해 설계되었습니다.
**Universal Inverted Bottleneck** (UIB)와 **Depthwise Separable Convolution**과 같은 혁신적인 기술을 결합하여 높은 정확도를 유지하면서도 계산 비용을 최소화하였으며, 모바일 디바이스나 키오스크와 같은 실시간 애플리케이션에 적합합니다.

---

#### **모델 구조**

1. **입력 (얼굴 이미지)**  
   - **입력 데이터:** 얼굴 이미지를 입력으로 처리하며, 얼굴 탐지 및 크기 조정을 통해 (224 × 224) 크기로 정규화합니다.  
   - **목표:** 입력 얼굴의 특징을 기반으로 인종 카테고리를 예측합니다.

2. **백본 네트워크 (MobileNetV4 Architecture)**  
   - **역할:** 입력 이미지를 처리하여 고차원 특징을 추출합니다.  
   - **구조:**  
     - **Universal Inverted Bottleneck (UIB):**  
       - Inverted Bottleneck, ConvNext, Feed Forward Network (FFN), ExtraDW와 같은 다양한 요소를 통합하여 특징 추출을 최적화합니다.  
       - 이러한 구조는 모델의 성능과 적응성을 크게 향상시킵니다.
     - **Mobile MQA (Mobile Multi-Head Attention):**  
       - 모바일 가속기에 최적화된 특별한 주의 메커니즘으로, 기존 주의 메커니즘 대비 39%의 속도 향상을 제공합니다.  
       - 이를 통해 모바일 하드웨어에서 추론 효율성을 극대화합니다.  
     - **Optimized Neural Architecture Search (NAS):**  
       - 최적화된 NAS 기법을 통해 다양한 모바일 플랫폼(CPU, DSP, GPU, Apple Neural Engine, Google Pixel EdgeTPU)에서 최적의 아키텍처를 생성합니다.
     - **Depthwise Separable Convolution:**  
       - 공간 및 채널 필터링을 분리하여 계산 효율성을 극대화합니다.  
   - **백본 출력:** 압축된 특징 맵을 생성하여 Fully Connected Layers로 전달합니다.

3. **Fully Connected Layers (FC Layers)**  
   - **특징:** 백본에서 추출된 특징 맵이 Fully Connected Layers로 전달됩니다.  
   - **출력:** FC Layers는 인종 카테고리의 클래스 확률을 예측합니다.

4. **예측 헤드 (Softmax Output)**  
   - **Softmax 분류:**  
     - 인종 카테고리(예: Asian, Non-Asian)에 대한 확률 분포를 생성합니다.  
     - 가장 높은 확률을 가진 카테고리를 최종 예측 값으로 지정합니다.

5. **제안된 손실 함수**  
   MobileNetV4는 **Cross Entropy Loss**를 사용하여 분류 오류를 최소화합니다:  
   - **Cross Entropy Loss:** 예측된 클래스 확률과 실제 레이블 간의 차이를 줄입니다.  
   - **Regularization (L2 Loss):** 네트워크의 큰 가중치를 규제하여 과적합을 방지합니다.  
   - **총 손실:**  $$\text{Total Loss} = \text{Cross Entropy Loss} + \text{L2 Regularization Loss}$$

---

#### **동작 원리 요약**
1. **입력 처리:** 입력된 얼굴 이미지를 네트워크에 맞게 크기 조정 및 정규화합니다.  
2. **특징 추출:** MobileNetV4의 경량화된 아키텍처를 통해 고차원 특징을 추출합니다.  
3. **예측 헤드:** Softmax를 사용하여 인종 카테고리를 분류합니다.  
4. **손실 함수 최적화:** Cross Entropy Loss와 정규화를 결합하여 정확도를 높이고 과적합을 방지합니다.

---

#### **모델 특징**
- **효율성:**  
  - Depthwise Convolution과 UIB를 활용한 경량 설계로 모바일 플랫폼에 적합합니다.  
  - Mobile MQA를 통해 기존 주의 메커니즘 대비 39% 빠른 추론 속도를 제공합니다.  
- **정확도:** 높은 분류 성능을 유지하면서도 계산량을 최소화합니다.  
- **실시간 애플리케이션:** 키오스크, 주의 모니터링, 인구 통계 연구와 같은 애플리케이션에 이상적입니다.  
- **최적화된 아키텍처:** NAS를 통해 다양한 플랫폼에서 최적의 성능을 보장합니다.  



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


<details>  
  <summary><h3> 얼굴 인식 - YOLOv11n-Face</h3></summary>  
  [YOLOv11n-Face GitHub](https://github.com/akanametov/yolo-face)<br>  

#### **구조 및 원리**  


##### **개요**  
YOLOv11n-Face는 **얼굴 탐지(Face Detection)** 및 **인식(Recognition)**을 위해 설계된 딥러닝 모델입니다. YOLO(You Only Look Once) 아키텍처를 기반으로 하며, 객체 탐지 기술의 발전을 활용하여 실시간 애플리케이션에 적합한 효율적이고 정확한 얼굴 탐지 솔루션을 제공합니다.

---

##### **모델 구조**  

###### 1. **입력 (Images or Video Frames)**  - **입력 데이터:**  
  - 모델은 이미지 또는 비디오 프레임을 입력으로 사용하며, 입력 데이터는 탐지에 최적화되도록 크기 조정 및 정규화 과정을 거칩니다.  
- **목표:**  
  - 입력 데이터에서 얼굴을 탐지하고 경계 상자(Bounding Box)의 좌표를 제공합니다.  

###### 2. **백본 네트워크 (YOLOv11n Architecture)**  - **역할:**  
  - YOLOv11n은 입력 이미지에서 특징을 추출합니다.  
- **구조:**  
  - **Convolutional Layers:** 입력 프레임에서 얼굴의 공간적 및 계층적 특징을 캡처합니다.  
  - **Neck Module:** 다양한 크기의 얼굴 탐지 정확도를 향상시키기 위해 특징 맵을 처리합니다.  
- **출력:**  
  - 다양한 크기의 얼굴을 탐지하기 위한 최적화된 특징 맵을 생성합니다.  

###### 3. **Detection Head**  - **Bounding Box Prediction:**  
  - 탐지된 얼굴에 대한 경계 상자 중심 좌표, 너비 및 높이를 예측합니다.  
- **Confidence Score:**  
  - 얼굴 존재 가능성을 나타내는 신뢰 점수를 할당합니다.  

###### 4. **후처리 (Post-Processing)**  - **Non-Maximum Suppression (NMS):**  
  - 중복된 경계 상자를 제거하고 가장 정확한 탐지를 유지합니다.  
- **최종 출력:**  
  - 탐지된 얼굴에 대한 경계 상자와 신뢰 점수 목록을 제공합니다.  

###### 5. **손실 함수 (Proposed Loss Function)**  
YOLOv11n-Face는 탐지 성능을 최적화하기 위해 다음 손실 함수를 사용합니다:  
- **Localization Loss:** 예측된 경계 상자 좌표와 실제 좌표 간의 차이를 줄입니다.  
- **Confidence Loss:** 정확한 신뢰 점수를 할당할 수 있도록 모델을 최적화합니다.  
- **총 손실:**  
  \[  
  \text{Total Loss} = \text{Localization Loss} + \text{Confidence Loss}  
  \]  

---

##### **동작 원리 요약**  1. **입력 처리:** 입력된 이미지 또는 비디오 프레임을 모델에 맞게 크기 조정 및 정규화합니다.  
2. **특징 추출:** YOLOv11n 백본을 통해 입력 데이터에서 고차원 특징을 추출합니다.  
3. **Detection Head:** 얼굴 탐지를 위해 경계 상자와 신뢰 점수를 예측합니다.  
4. **후처리:** Non-Maximum Suppression(NMS)을 사용하여 탐지 결과를 정제합니다.  
5. **손실 최적화:** Localization Loss와 Confidence Loss를 결합하여 탐지 정확도를 높입니다.  

---

##### **모델 특징**  - **효율성:** YOLOv11n은 실시간 얼굴 탐지 애플리케이션에 최적화되어 있습니다.  
- **정확성:** 까다로운 환경에서도 얼굴 탐지의 정밀도와 재현율을 균형 있게 유지합니다.  
- **적용 가능성:** 감시, 인증, 증강 현실(AR)과 같은 다양한 얼굴 탐지 애플리케이션에 적합합니다.  

</details>


<details>
  <summary><h3>6. 시선추정 - L2CS-Net</h3></summary>
  [L2CS-Net Paper](https://arxiv.org/abs/2203.03339)<br>
  [L2CS-Net GitHub](https://github.com/Ahmednull/L2CS-Net)<br>
  [L2CS-Net roboflow](https://blog.roboflow.com/gaze-direction-position/)<br>
  
  #### 구조 및 원리
  ![image](https://github.com/user-attachments/assets/b7573e01-bcce-4b90-b3ca-e7a149caf802)
  
  ##### 개요
  L2CS-Net은 **시선 추정(Gaze Estimation)**을 위한 딥러닝 기반 네트워크로, 입력 얼굴 이미지를 이용하여 사람의 수평(Yaw) 및 수직(Pitch) 방향의 시선을 정확히 예측하는 모델입니다. 이 모델은 **분류(Classification)**와 **회귀(Regression)** 기법을 결합하여 시선 추정 정확도와 효율성을 극대화합니다.
  
  ##### 모델 구조
  
  ###### 1. **입력 (Face Images)**
  - **입력 데이터:**
    - 얼굴 이미지를 네트워크에 입력으로 사용하며, 얼굴 탐지 및 정규화를 통해 전처리된 이미지입니다.
  - **목표:**
    - 입력 얼굴에서 수평(Yaw) 및 수직(Pitch) 방향 시선을 추정합니다.
  
  ###### 2. **백본 네트워크 (ResNet-50 Backbone)**
  - **역할:**
    - ResNet-50은 입력 이미지에서 중요한 고차원 특징을 추출합니다.
  - **구조:**
    - ResNet-50은 합성곱(CNN) 레이어로 구성되어 있으며, 입력 이미지의 공간적 패턴을 학습합니다.
  - **출력:**
    - ResNet-50은 추출된 특징을 Fully Connected Layers로 전달합니다.
  
  ###### 3. **Fully Connected Layers (FC Layers)**
  - ResNet-50에서 추출된 특징은 두 개의 Fully Connected Layer로 전달됩니다.
  - 두 FC Layer는 각각 **Yaw(수평)**와 **Pitch(수직)** 방향의 시선을 예측합니다.
  
  ###### 4. **Yaw 및 Pitch 헤드 (Multi-Head Output)**
  4.1 **Yaw 헤드**
  - **Softmax 분류:** 수평 각도 클래스(예: -30°, -15°, 0°, 15°, 30° 등)에 대해 확률 분포를 계산합니다.
  - **Expectation 계산:** Softmax 확률을 기반으로 각 클래스의 기대값을 계산하여 연속적인 각도를 예측합니다.
  
  4.2 **Pitch 헤드**
  - **Softmax 분류:** 수직 각도 클래스(예: -30°, -15°, 0°, 15°, 30° 등)에 대해 확률 분포를 계산합니다.
  - **Expectation 계산:** Softmax 확률을 기반으로 각 클래스의 기대값을 계산하여 연속적인 각도를 예측합니다.
  
  ###### 5. **손실 함수 (Proposed Loss Function)**
  L2CS-Net은 **분류 손실**과 **회귀 손실**을 결합하여 학습합니다:
  
  5.1 **Yaw 손실**
  - **크로스 엔트로피 손실 (Cross Entropy Loss):** 클래스 확률과 실제 각도 클래스 간 차이를 줄입니다.
  - **평균 제곱 오차 (Mean Squared Error, MSE):** 계산된 기대값과 실제 연속값 간의 오차를 줄입니다.
  - **총 손실:**
    \[
    \text{Total Yaw Error} = \text{Cross Entropy Loss} + \text{Mean Squared Error}
    \]
  
  5.2 **Pitch 손실**
  - Pitch 손실은 Yaw 손실과 동일한 방식으로 계산됩니다:
    \[
    \text{Total Pitch Error} = \text{Cross Entropy Loss} + \text{Mean Squared Error}
    \]
  
  ##### 동작 원리 요약
  1. **입력 처리:** 얼굴 이미지를 네트워크에 입력.
  2. **특징 추출:** ResNet-50 백본을 통해 고차원 특징을 추출.
  3. **Yaw 및 Pitch 헤드:** 각각 수평(Yaw) 및 수직(Pitch) 방향에 대해 Softmax와 기대값 계산으로 각도 예측.
  4. **손실 함수 최적화:** 분류 손실과 회귀 손실을 결합하여 모델 학습.
  
  ##### 모델의 특징
  - **효율성:** ResNet-50을 백본으로 사용하여 고차원 특징 추출.
  - **정확성:** 분류와 회귀를 결합하여 연속적인 각도를 예측.
  - **적용 가능성:** 다양한 시선 추정 응용 분야(예: AR/VR, 주의 모니터링, 졸음 감지)에 적합.
</details>



-- 각 최종모델에 대한 상세 설명 추가 (원리  ,구조 )
---
# 시선추적 관련 calibration 생성
https://github.com/kdt-kiosk/kiosk_gaze



--- 
# UI

일반 UI
![image](https://github.com/user-attachments/assets/3c695953-b2d3-4f2b-a2f5-5f30f6ad819f)


Senior UI
![image](https://github.com/user-attachments/assets/20beeb36-6db0-4735-af55-3354d00bb5cf)

외국인으로 인식될시 메뉴화면대신 언어선택창이 먼저 출력
![image](https://github.com/user-attachments/assets/b3e526a8-38bb-42e7-aef8-d3824decc0b6)



# 시연 동영상

** 40세 이상인 아닌 사람이 일반 UI로 이용하는 모습
분류모델로 이용해서 추천메뉴를 사용자의 상태에 맞게 설정가능 




https://github.com/user-attachments/assets/f4172814-cdc7-4af5-85de-da0e1a692c96  


** 40세 이상인  사람이  전용UI로 시선추적으로 이용하는 모습


https://github.com/user-attachments/assets/54ce00b6-655c-4a77-b058-c0359bc9db30


# 결과
AI 모델을 이용해서 사용자 맞춤 서비스를 이용할수있다
취약층 계층에게 시선으로만 주문이 가능하게 만들어 접근성을 높인다. <- 이런식으로 작성 Backgroud 참고


