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
  <summary><h1> 감정분류</h1></summary>
  <!-- 내용 -->
</details>


## 인종분류

## 성별분류

## 연령분류

만약 토글이 안된다면 따로 페이지 생성해서 작성
----

ex) 분류모델별 성능비교
https://cdn.discordapp.com/attachments/1229239889799807042/1313442062179504200/image.png?ex=675025be&is=674ed43e&hm=d8e88e70b6580a5ff28bbe358bb6eef2e32a9aa447ef3efcab8585059d6f32dd&


~~ 선택된 최종모델
## 최종사용모델
감정분류 [DDAMNet](https://github.com/SainingZhang/DDAMFN)   
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


