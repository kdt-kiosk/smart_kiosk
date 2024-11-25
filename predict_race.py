## 현재 model이 최신것이 아니라 이상하게 예측되는듯함 집에가서 모델 최신거로 받아서 확인해보기.

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# VGG16 기반 인종 분류 모델 로드 및 가중치 적용 함수
def load_custom_vgg16_model(drop_rate=0.2):
    model = models.vgg16(pretrained=False)
    
    # 모든 파라미터의 업데이트를 비활성화하고 마지막 두 레이어만 활성화
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier[-2:].parameters():
        param.requires_grad = True
    
    # 드롭아웃을 포함한 분류 레이어 설정
    model.classifier = nn.Sequential(
        nn.Dropout(p=drop_rate),  # 지정된 드롭아웃 비율 사용
        nn.Linear(25088, 4096),
        nn.ReLU(),
        nn.Dropout(p=drop_rate),
        nn.Linear(4096, 2)  # 최종 이진 분류 레이어 (oriental과 others)
    )
    
    # 학습된 모델 가중치 로드
    model.load_state_dict(torch.load('models/1108_race_vgg16_model.pth', map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
    model.eval()  # 평가 모드로 설정
    return model

# 이미지 전처리 설정
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # VGG16 입력 크기로 조정
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 인종 예측 함수
def predict_race(image_path):
    model = load_custom_vgg16_model()  # 모델 로드
    
    # 이미지 열기 및 전처리
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # 배치 차원 추가
    
    # 예측 수행
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    print('.py에서의 race눈',predicted.item())
    # 0이면 oriental, 1이면 others로 가정
    return "oriental" if predicted.item() == 0 else "other"
