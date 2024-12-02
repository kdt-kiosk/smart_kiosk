import joblib
import numpy as np
import cv2
import base64
import requests
from dotenv import load_dotenv
import os

# .env 파일 로드
load_dotenv()
# Docker API 설정
API_KEY = os.getenv("API_KEY")
GAZE_DETECTION_URL = os.getenv("GAZE_DETECTION_URL")
file_path_x_pkl = "./calibration/x_model_web.pkl"
file_path_y_pk = "./calibration/y_model_web.pkl"

# 시선 보정
def calibraion(predicted_x, predicted_y):
    global file_path_x_pkl, file_path_y_pk
    x_model = joblib.load(file_path_x_pkl)
    y_model = joblib.load(file_path_y_pk)
    corrected_x = np.rint(x_model.predict(np.array(predicted_x).reshape(-1, 1))).astype(int)
    corrected_y = np.rint(y_model.predict(np.array(predicted_y).reshape(-1, 1))).astype(int)
    return corrected_x[0], corrected_y[0]

# 시선 추정
def detect_gazes(frame: np.ndarray):
    img_encode = cv2.imencode(".jpg", frame)[1]
    img_base64 = base64.b64encode(img_encode)
    resp = requests.post(
        GAZE_DETECTION_URL,
        json={
            "api_key": API_KEY,
            "image": {"type": "base64", "value": img_base64.decode("utf-8")},
        },
    )
    gazes = resp.json()[0]["predictions"]
    return gazes

# 화면 크기 조정(전체화면)
def resize_with_padding(frame, target_width, target_height, padding_color=(0, 0, 0)):
    """
    원본 프레임을 타겟 크기에 맞게 리사이즈하면서 비율을 유지하고, 패딩 추가.

    :param frame: 원본 프레임
    :param target_width: 목표 가로 크기
    :param target_height: 목표 세로 크기
    :param padding_color: 패딩 색상 (기본값: 검정색)
    :return: 패딩이 추가된 새로운 프레임
    """
    # 원본 크기
    original_height, original_width = frame.shape[:2]

    # 비율 계산
    ratio_w = target_width / original_width
    ratio_h = target_height / original_height
    scale = min(ratio_w, ratio_h)  # 비율을 유지하면서 화면에 맞추기 위한 스케일

    # 새 크기 계산
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    # 프레임 리사이즈
    resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # 배경 생성
    canvas = np.full((target_height, target_width, 3), padding_color, dtype=np.uint8)

    # 화면 중심에 삽입
    top = (target_height - new_height) // 2
    left = (target_width - new_width) // 2
    canvas[top:top + new_height, left:left + new_width] = resized_frame

    return canvas
