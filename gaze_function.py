import joblib
import numpy as np
import cv2
import base64
import requests
from dotenv import load_dotenv
import os
import pickle

# .env 파일 로드
load_dotenv()
# Docker API 설정
API_KEY = os.getenv("API_KEY")
GAZE_DETECTION_URL = os.getenv("GAZE_DETECTION_URL")
file_path_x_pkl = "./calibration/x_model_web.pkl"
file_path_y_pk = "./calibration/y_model_web.pkl"

# 카메라 왜곡 보정
def undistort_frame(frame, calibration_file_path='distortion_correction.pkl'):
    """
    주어진 프레임의 왜곡을 제거하는 함수.
    
    Args:
        frame (numpy.ndarray): 입력 프레임 (웹캠 등에서 캡처된 이미지).
        calibration_file_path (str): 카메라 왜곡 보정 데이터를 저장한 파일 경로. 
                                     기본값은 'distortion_correction.pkl'.
    
    Returns:
        numpy.ndarray: 왜곡이 제거된 프레임.
    """
    # calibration 파일로 경로 고정
    calibration_path = os.path.join("calibration", calibration_file_path)

    # 파일이 존재하지 않으면 입력 프레임 그대로 반환
    if not os.path.exists(calibration_path):
        print(f"왜곡 보정 파일을 찾을 수 없습니다. 원본 프레임을 반환합니다: {calibration_path}")
        return frame

    # 왜곡 보정 데이터 불러오기
    with open(calibration_path, 'rb') as f:
        data = pickle.load(f)
    mtx = data['mtx']
    dist = data['dist']
    
    # 입력 프레임 크기 가져오기
    h, w = frame.shape[:2]
    
    # 새로운 카메라 매트릭스 계산
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    
    # 왜곡 보정
    undistorted_frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)
    
    ## # ROI를 사용하여 검은 영역 제거 (선택 사항)
    ## # 왜곡 제거 후 남는 검은 영역을 제거하기 위해 ROI를 사용하여 이미지 크롭. 적용하면 원본과 크기가 달라지니 유의
    # x, y, w, h = roi
    # undistorted_frame = undistorted_frame[y:y+h, x:x+w]
    
    return undistorted_frame

# 시선 보정
def calibraion(predicted_x, predicted_y):
    '''
    주어진 x, y 좌표(예측값)를 사전 학습된 모델을 사용해 보정하여 정확한 시선 좌표를 반환합니다.

    매개변수:
        - predicted_x (float or int): 예측된 x 좌표.
        - predicted_y (float or int): 예측된 y 좌표.
    글로벌 변수:
        - file_path_x_pkl: x 좌표 보정 모델이 저장된 .pkl 파일 경로.
        - file_path_y_pk: y 좌표 보정 모델이 저장된 .pkl 파일 경로.
    작동 과정:
        joblib.load()를 사용하여 x와 y 좌표에 대한 보정 모델을 각각 로드합니다.
        예측된 x와 y 좌표를 각각 모델에 입력하여 보정된 좌표를 계산합니다.
        np.rint()로 보정된 좌표를 반올림하고 정수로 변환합니다.
        보정된 x와 y 좌표를 반환합니다.
    반환값:
        - (int, int): 보정된 x, y 좌표.
    '''
    global file_path_x_pkl, file_path_y_pk
    x_model = joblib.load(file_path_x_pkl)
    y_model = joblib.load(file_path_y_pk)
    corrected_x = np.rint(x_model.predict(np.array(predicted_x).reshape(-1, 1))).astype(int)
    corrected_y = np.rint(y_model.predict(np.array(predicted_y).reshape(-1, 1))).astype(int)
    return corrected_x[0], corrected_y[0]

# 시선 추정
def detect_gazes(frame: np.ndarray):
    '''
    주어진 이미지 프레임에서 시선을 추정하여 좌표값을 반환합니다. 외부 API를 사용합니다.
    매개변수:
        frame (numpy.ndarray): 웹캠 등에서 캡처된 이미지 프레임.
    작동 과정:
        프레임을 JPEG 형식으로 인코딩합니다.
        인코딩된 이미지를 Base64 형식으로 변환합니다.
        HTTP POST 요청을 통해 시선 추정 API를 호출합니다. 요청 데이터는 다음과 같습니다:
        API 키: API_KEY를 사용하여 인증.
        이미지 데이터: Base64로 인코딩된 이미지.
        API 응답에서 시선 추정 좌표(predictions)를 추출합니다.
    반환값:
        gazes (list): 시선 추정 결과를 포함하는 리스트. 각 항목은 시선 좌표 정보를 포함.
    '''
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
