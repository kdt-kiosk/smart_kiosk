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



## 인종분류

<details>
  <summary><h2> 성별분류</h2></summary>
  
  ## 1. Mini-Xception 모델 (사전학습된 모델을 fine-tuning 및 tranfer Learning)
  ### 변환된 Markdown 및 Key Insights
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

## 연령분류


만약 토글이 안된다면 따로 페이지 생성해서 작성
----

ex) 분류모델별 성능비교
https://cdn.discordapp.com/attachments/1229239889799807042/1313442062179504200/image.png?ex=675025be&is=674ed43e&hm=d8e88e70b6580a5ff28bbe358bb6eef2e32a9aa447ef3efcab8585059d6f32dd&


~~ 선택된 최종모델
## 최종사용모델
### 1. 감정분류 - DDAMNet
[DDAMNet GitHub](https://github.com/SainingZhang/DDAMFN)<br>
[DDAMNet Paper](https://www.mdpi.com/2079-9292/12/17/3595)
### 구조 및 원리
![image](https://github.com/user-attachments/assets/b4ec1c20-1452-4018-a310-81e92cfd9a08)
#### 개요
DDAMFN(Dual-Direction Attention Mixed Feature Network)은 얼굴 표정 인식(Facial Expression Recognition, FER)을 위해 설계된 경량화되고 강력한 딥러닝 모델입니다. 이 네트워크는 입력 이미지로부터 특징을 추출하고, 중요한 부분에 집중하여 얼굴 표정을 정확하게 예측합니다. DDAMFN은 크게 두 가지 주요 구성 요소로 나뉩니다:
1. **MFN (Mixed Feature Network):** 얼굴 이미지로부터 기본적인 특징을 추출.
2. **DDAN (Dual-Direction Attention Network):** 추출된 특징을 분석하고, 얼굴에서 중요한 부분에 주의를 집중.

---

#### 모델 구조

##### 1. **Feature Extraction (특징 추출)**
- 입력 이미지 크기: \(112 \times 112\)
- **MFN (Mixed Feature Network):**
  - 다양한 크기의 합성곱 커널(MixConv)을 사용하여 얼굴 이미지에서 세부적이고 전반적인 특징을 추출합니다.
  - Residual Bottleneck 구조를 통해 정보 손실을 방지하고, 효율적인 학습이 가능하도록 설계되었습니다.
  - Stride-1과 Stride-2 블록을 활용하여 세부 정보와 주요 특징을 모두 보존합니다.
  - **Coordinate Attention**을 적용하여 얼굴 이미지의 중요한 위치(눈, 코, 입 등)를 강조합니다.
  - 최종적으로 \(7 \times 7 \times 512\) 크기의 특징 맵을 생성합니다.

---

##### 2. **Attention Module (주의 모듈)**
- **Dual Direction Attention (DDA):**
  - 두 가지 방향으로 주의를 분석:
    - **수평 방향(X Linear GDConv):** 얼굴의 좌우 영역을 분석.
    - **수직 방향(Y Linear GDConv):** 얼굴의 위아래 영역을 분석.
  - 두 방향의 결과를 결합(Concatenate & Conv2D)하여 종합적인 주의 맵을 생성합니다.
- **주의 맵 생성:** 중요한 부분을 강조하고, 덜 중요한 영역의 영향을 줄입니다.
- **MAX 연산:** 여러 주의 맵 중에서 가장 유용한 맵을 선택합니다.
- **곱셈 연산:** 주의 맵과 기존 특징 맵을 결합하여, 모델이 얼굴의 중요한 영역에 집중할 수 있도록 합니다.

---

##### 3. **Feature Transformation (특징 변환)**
- **Global Depthwise Convolution (GDConv):**
  - 주의 맵이 결합된 특징 맵을 압축하여 \(1 \times 1 \times 512\) 크기로 변환합니다.
- **Reshape:** 특징 맵을 256차원의 벡터로 변환하여 최종 예측에 사용합니다.

---

##### 4. **Fully Connected Layer (완전 연결 계층)**
- 변환된 256차원 벡터는 완전 연결 계층(FC Layer)을 통과하여, 입력 얼굴 이미지의 감정을 예측합니다.
- **손실 함수(Loss Function):**
  - **Cross-Entropy Loss (\(L_{cls}\)):** 모델의 감정 예측 성능을 최적화합니다.
  - **Attention Loss (\(L_{att}\)):** 주의 모듈이 서로 다른 얼굴 영역에 집중하도록 유도합니다.

---

#### 동작 원리 요약
1. **MFN (특징 추출):** 얼굴 이미지로부터 다양한 크기의 커널을 활용해 세부적이고 전반적인 특징을 추출.
2. **DDAN (주의 모듈):** 두 방향(수평, 수직)으로 얼굴의 중요한 영역을 분석하고, 주의 맵을 생성.
3. **Feature Transformation:** 추출된 특징을 압축하고 변환하여 예측 가능한 벡터로 변환.
4. **FC Layer:** 최종 예측을 통해 얼굴 감정을 분류.

---

#### 모델의 강점
1. **경량화:** 계산량을 줄이면서도 높은 정확도를 유지.
2. **효율적인 특징 추출:** 다양한 크기의 커널과 Residual Bottleneck 구조를 활용.
3. **주의 메커니즘:** 얼굴의 중요한 부분에 집중하여 예측 성능 향상.
4. **적용 가능성:** 다양한 감정 인식 및 얼굴 기반 응용 프로그램에 활용 가능.




인종분류 [mobilenetV4](https://huggingface.co/blog/rwightman/mobilenetv4)  
성별분류 [resnet18](https://huggingface.co/docs/transformers/model_doc/resnet)  
연령분류 [resnet18](https://huggingface.co/docs/transformers/model_doc/resnet)  

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


