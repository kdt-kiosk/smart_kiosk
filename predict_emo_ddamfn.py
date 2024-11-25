from torch import nn
import torch
#from networks.DDAM import DDAMNet
from torchvision import models, transforms
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, Sequential, Module
from PIL import Image




# 이미지 전처리 설정
transform = transforms.Compose([
        transforms.Resize((112, 112)),  # 이미지 크기 조정
    transforms.ToTensor(),  # 이미지를 텐서로 변환
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 정규화
])
# 감정 예측 함수
def predict_emo(image_path):
    # 감정 분류 모델 로드
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = 'models/emotion_DDAMFN_model.pth'
    model = torch.load(model_path, map_location=device)



    # 이미지 열기 및 전처리
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # 배치 차원 추가

   

    print("Preprocessed image shape:", image.shape)  # 이미지 크기 확인

    # 예측 수행
    with torch.no_grad():
        image = image.to(device)
        output = model(image)  # 예측
        # 첫 번째 요소만 사용 (최종 클래스별 로짓)
        logits = output[0]  # 최종 예측 결과
        predicted_class = torch.argmax(logits, dim=1).item()  # 가장 높은 확률의 클래스 선택

    # 예측 결과 매핑
    predicted_emotion = emotion_mapping(predicted_class)
    return predicted_emotion


# 감정 매핑 함수
def emotion_mapping(emotion):
    emotion_dict = {
        0: "부정",  # 부정적 감정: angry, disgust, fear, sad
        1: "긍정",  # 긍정적 감정: happy
        2: "중립"   # 중립적 감정: neutral, surprise
    }
    return emotion_dict.get(emotion, "Unknown")
