import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# 커스텀 레이어가 포함된 ResNet18 모델 로드 함수
def load_custom_resnet18_model():
    res18model = models.resnet18(pretrained=False)
    num_ftrs = res18model.fc.in_features
    
    # 학습한 커스텀 레이어 구성
    layers = [
        nn.Dropout(0.3),
        nn.Linear(num_ftrs, 1024),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 3)  # 3개의 클래스가 있다고 가정
    ]
    res18model.fc = nn.Sequential(*layers)

    # 모델 로드 (strict=False)
    model_path = os.path.join('models', 'age_resnet18_model.pth')
    res18model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)
    res18model.eval()
    return res18model

# 이미지 전처리 설정
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 성별 예측 함수
def predict_age(image_path):
    model = load_custom_resnet18_model()  # 변경된 함수 이름으로 호출
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # 배치 차원 추가
    
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    print('0이면 20세이하, 1이면 20~39세, 2이면 40세 이상',predicted.item())
    # 예측에 따라 성별을 반환
    if predicted.item() == 0:
        return "20세 이하"
    elif predicted.item() == 1:
        return "20~39세"
    elif predicted.item() == 2:
        return "40세 이상"
    else:
        return "알 수 없음"  # 예외 처리
