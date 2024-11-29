from flask import Flask, render_template, Response, redirect, url_for, session,request,jsonify,Blueprint
import cv2
import os
import json
import asyncio
import threading
import time
from ultralytics import YOLO  # yolov8 모델 불러오기 (yolov8 face용 커스텀 설정 필요)
from predict_gender import predict_gender
from predict_age import predict_age
from predict_race_mobilenetv4 import predict_race  # 인종 예측 함수가 있다고 가정
from predict_emo_ddamfn import predict_emo


app = Flask(__name__)
app.secret_key = "your_secret_key"
# 모든 출처에서 CORS 허용 (보안상 특정 도메인으로 제한하는 것이 더 안전)


# Blueprint 생성
senior = Blueprint('senior', __name__, url_prefix='/senior')

# 이미지 폴더 경로 설정
IMAGE_FOLDER = 'static/images'
# 카테고리별로 폴더에서 이미지 읽기

def load_images_from_folder(category, start_id=1):
    folder_path = os.path.join(IMAGE_FOLDER, category.lower())
    items = []
    for idx, filename in enumerate(os.listdir(folder_path)):
        item_name = os.path.splitext(filename)[0]  # 파일명에서 확장자 제거
        items.append({
            "id": start_id + idx,
            "name": item_name,  # 번역되지 않은 원래 이름
            "price": round(1.5 + len(items) * 0.5, 2),
            "category": category,
            "image": f"images/{category.lower()}/{filename}"
        })
    return items

# 메뉴 항목 로드 (카테고리별 ID 구분을 위해 start_id를 조정)
menu_items = load_images_from_folder("tea", start_id=1) \
           + load_images_from_folder("coffee", start_id=100) \
           + load_images_from_folder("beverage", start_id=200)
# 한 페이지에 표시할 메뉴 수
ITEMS_PER_PAGE = 12
SENIOR_ITEMS_PER_PAGE = 4
redirect_url = None  # 리디렉션 대상 URL


UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # 저장 폴더가 없으면 생성

# yolov8 모델 로드
face_model = YOLO("models/yolov11n-face.pt")  # yolov8 face 모델 경로 지정
cap = cv2.VideoCapture(0)  # 웹캠 시작

# 얼굴 인식 변수
face_detected = False
face_detected_time = 0

        
def generate_frames():
    global face_detected, face_detected_time, redirect_url
    redirect_url = None

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)  # 좌우 반전 처리

        # YOLO 모델로 얼굴 감지
        results = face_model.predict(frame, stream=True)
        face_in_frame = False
        cropped_face = None

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # 얼굴 감지 영역 표시
                face_in_frame = True
                cropped_face = frame[y1:y2, x1:x2]

        # 얼굴 인식 시간 계산 및 리디렉션 처리
        if face_in_frame:
            if not face_detected:
                face_detected = True
                face_detected_time = time.time()
            elif time.time() - face_detected_time >= 2:
                if cropped_face is not None:
                    save_path = os.path.join(UPLOAD_FOLDER, 'test_image.jpg')
                    cv2.imwrite(save_path, cropped_face)

                    # 나이 분류 수행
                    age = predict_age(save_path)

                    # 리디렉션 URL 설정
                    redirect_url = '/senior/' if age != '40세 이상' else '/home'
                    break
        else:
            face_detected = False
            face_detected_time = None


        # 프레임을 JPEG 형식으로 변환
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')





# 첫 화면(카메라 화면) 라우트
@app.route('/')
def index():
    global redirect_url
    redirect_url = None  # 초기화
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    global redirect_url
    print('redirect_irl은??',redirect_url)
    if redirect_url:
        return f"REDIRECT:{redirect_url}"
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# 홈 페이지 라우트 (기존 '/'을 '/home'으로 변경)
@app.route('/home')
def home():
    category = request.args.get('category')
    page = int(request.args.get('page', 1))

    test_image_path = os.path.join('uploads', 'test_image.jpg')
    if os.path.exists(test_image_path):
        # 성별과 인종 예측 수행
        gender = predict_gender(test_image_path)
        race = predict_race(test_image_path)
        age = predict_age(test_image_path)
        
        # 성별을 세션에 저장하여 recommend 라우트에서 사용
        session['gender'] = gender
        session['age'] = age

        # 인종이 'other'이고 언어 선택이 아직 설정되지 않았다면 모달 표시
        if race == 'other' and not session.get('language_selected'):
            language_modal = True
            session['language_selected'] = True  # 모달을 한 번만 표시하도록 설정
        else:
            language_modal = False
    else:
        gender,age, race, language_modal = None, None,None, False

    # 카테고리 필터링
    if category in ['coffee', 'tea', 'beverage']:
        filtered_menu = [item for item in menu_items if item["category"] == category]
    else:
        filtered_menu = menu_items

    start = (page - 1) * ITEMS_PER_PAGE
    end = start + ITEMS_PER_PAGE
    paginated_menu = filtered_menu[start:end]
    total_pages = (len(filtered_menu) - 1) // ITEMS_PER_PAGE + 1

    cart = session.get("cart", [])
    total_price = sum(item["price"] * item["quantity"] for item in cart if "quantity" in item)

    return render_template(
        'home.html',
        menu=paginated_menu,
        cart=cart,
        total_price=total_price,
        page=page,
        total_pages=total_pages,
        category=category,
        language_modal=language_modal
    )

# 추천 페이지 라우트
@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    test_image_path = os.path.join('uploads', 'test_image.jpg')
    cart = session.get("cart", [])
    total_price = sum(item["price"] * item["quantity"] for item in cart)

    # 성별 및 감정 예측 수행
    if os.path.exists(test_image_path):
        gender = predict_gender(test_image_path)
        age = predict_age(test_image_path)
        emotion = predict_emo(test_image_path)  # 감정 예측 수행
        session['gender'] = gender  # 세션에 성별 저장
    else:
        gender = session.get('gender', 'female')  # 기본값을 'female'로 설정
        emotion = "중립"  # 기본값으로 "중립" 감정 설정

    # 성별에 따라 추천 메뉴 필터링
    if gender == 'male':
        recommended_menu = [item for item in menu_items if item["category"] == "coffee"]
    else:
        recommended_menu = [item for item in menu_items if item["category"] in ["tea", "beverage"]]
    current_language = session.get('language', 'ko')
    
    # 디버깅: gender와 emotion 값 출력
    print("Predicted gender, emotion:", gender,emotion)
    print("current_language:", current_language)
    print("Recommended menu items:", [item["name"] for item in recommended_menu])
    
    return render_template(
        'recommend.html',
        menu=recommended_menu,
        cart=cart,
        total_price=total_price,
        gender=gender,  # gender 변수를 템플릿에 전달
        age=age,  # age 변수를 템플릿에 전달
        emotion=emotion,  # emotion 변수를 템플릿에 전달
        category='recommend',  # 'recommend' 페이지임을 템플릿에 전달
        current_language=current_language
    )





# 카트에 아이템 추가 라우트
@app.route('/add_to_cart/<int:item_id>', methods=['POST'])
@senior.route('/add_to_cart/<int:item_id>', methods=['POST'])
def add_to_cart(item_id):
    item = next((item for item in menu_items if item["id"] == item_id), None)
    if item:
        if "cart" not in session:
            session["cart"] = []
        cart = session["cart"]

        # 카트에 같은 항목이 있을 경우 수량을 증가
        existing_item = next((cart_item for cart_item in cart if cart_item["id"] == item_id), None)
        if existing_item:
            existing_item["quantity"] += 1
        else:
            cart.append({"id": item["id"], "name": item["name"], "price": item["price"], "quantity": 1})

        session.modified = True
        current_language = session.get('language', 'ko')
        # 장바구니 HTML을 다시 렌더링하여 JSON 응답으로 반환
        cart_html = render_template('cart_partial.html', cart=cart, total_price=sum(item["price"] * item["quantity"] for item in cart),
                                    current_language=current_language)
        return jsonify(success=True, cart_html=cart_html)
    return jsonify(success=False)

# 아이템 수량 증가감소 라우트
@app.route('/update_quantity/<int:item_id>/<string:action>', methods=['POST'])
@senior.route('/update_quantity/<int:item_id>/<string:action>', methods=['POST'])
def update_quantity(item_id, action):
    cart = session.get("cart", [])
    
    for item in cart:
        if item["id"] == item_id:
            if action == 'increase':
                item["quantity"] += 1
            elif action == 'decrease':
                if item["quantity"] > 1:
                    item["quantity"] -= 1
                else:
                    cart.remove(item)
            break
    
    session["cart"] = cart
    session.modified = True
    # 세션에서 언어 가져오기
    current_language = session.get('language', 'ko')

    # 새롭게 렌더링된 카트 HTML을 반환
    cart_html = render_template('cart_partial.html', cart=cart, total_price=sum(item["price"] * item["quantity"] for item in cart),
                       current_language=current_language)
    return jsonify({"success": True, "cart_html": cart_html})

# 아이템 삭제 라우트
@app.route('/remove_item/<int:item_id>', methods=['POST'])
@senior.route('/remove_item/<int:item_id>', methods=['POST'])
def remove_item(item_id):
    cart = session.get("cart", [])
    session["cart"] = [item for item in cart if item["id"] != item_id]
    session.modified = True

    # 새롭게 렌더링된 카트 HTML을 반환
    cart_html = render_template('cart_partial.html', cart=session["cart"], total_price=sum(item["price"] * item["quantity"] for item in session["cart"]))
    return jsonify({"success": True, "cart_html": cart_html})

# 카트 페이지 라우트
@app.route('/cart')
@senior.route('/cart')
def cart():
    cart = session.get("cart", [])
    total_price = sum(item["price"] * item["quantity"] for item in cart)
    return render_template('cart.html', cart=cart, total_price=total_price)

# 새로운 체크아웃 라우트
@app.route('/checkout')
def checkout():
    session.pop("cart", None)  # 카트 비우기
    return redirect(url_for('home'))  # 메인 페이지로 리디렉션


## 여기서부터 추가중
# 시니어홈 페이지 라우트
@senior.route('/')
def home():
    # 선택된 카테고리와 페이지 번호를 가져옵니다.
    category = request.args.get('category')
    page = int(request.args.get('page', 1))  # 기본 페이지 번호는 1
    
    # 카테고리에 따라 메뉴 필터링
    if category:
        filtered_menu = [item for item in menu_items if item["category"] == category]
    else:
        filtered_menu = menu_items  # 카테고리가 없으면 전체 메뉴 표시
    
    # 페이지네이션 처리
    start = (page - 1) * SENIOR_ITEMS_PER_PAGE
    end = start + SENIOR_ITEMS_PER_PAGE
    paginated_menu = filtered_menu[start:end]
    total_pages = (len(filtered_menu) + SENIOR_ITEMS_PER_PAGE - 1) // SENIOR_ITEMS_PER_PAGE
    # page가 범위 안에 있도록 설정
    if page < 1:
        page = 1
    elif page > total_pages:
        page = total_pages

    # 카트 정보 가져오기
    cart = session.get("cart", [])
    total_price = sum(item["price"] * item["quantity"] for item in cart if "quantity" in item)
    print('토탈 페이지는',total_pages,page)
    print('아이템은',paginated_menu)
    # home.html 템플릿으로 렌더링
    return render_template(
        'senior_home.html',
        menu=paginated_menu,
        cart=cart,
        total_price=total_price,
        page=page,
        total_pages=total_pages,
        category=category  # 현재 카테고리를 전달
    )
@senior.route('/recommend', methods=['GET', 'POST'])
def recommend():
    test_image_path = os.path.join('uploads', 'test_image.jpg')
    cart = session.get("cart", [])
    total_price = sum(item["price"] * item["quantity"] for item in cart)
    page = int(request.args.get('page', 1))  # 기본 페이지 번호는 1
    # 성별 및 감정 예측 수행
    if os.path.exists(test_image_path):
        gender = predict_gender(test_image_path)
        age = predict_age(test_image_path)
        emotion = predict_emo(test_image_path)  # 감정 예측 수행
        session['gender'] = gender  # 세션에 성별 저장
    else:
        gender = session.get('gender', 'female')  # 기본값을 'female'로 설정
        emotion = "중립"  # 기본값으로 "중립" 감정 설정

    # 성별에 따라 추천 메뉴 필터링
    if gender == 'male':
        recommended_menu = [item for item in menu_items if item["category"] == "coffee"]
    else:
        recommended_menu = [item for item in menu_items if item["category"] in ["tea", "beverage"]]
     # 페이지네이션 처리
    start = (page - 1) * SENIOR_ITEMS_PER_PAGE
    end = start + SENIOR_ITEMS_PER_PAGE
    paginated_menu = recommended_menu[start:end]
    total_pages = (len(recommended_menu) + SENIOR_ITEMS_PER_PAGE - 1) // SENIOR_ITEMS_PER_PAGE
    # page가 범위 안에 있도록 설정
    if page < 1:
        page = 1
    elif page > total_pages:
        page = total_pages
    
    # 디버깅: gender와 emotion 값 출력
    print("Predicted gender, emotion:", gender,emotion)
    #print("Predicted emotion:", emotion)
    print("Recommended menu items:", [item["name"] for item in recommended_menu])
    current_language = session.get('language', 'ko')
    return render_template(
        'senior_recommend.html',
        menu=paginated_menu,
        cart=cart,
        total_price=total_price,
        page=page,
        total_pages=total_pages,
        gender=gender,  # gender 변수를 템플릿에 전달
        age=age,  # age 변수를 템플릿에 전달
        emotion=emotion,  # emotion 변수를 템플릿에 전달
        category='recommend',  # 'recommend' 페이지임을 템플릿에 전달
        current_language=current_language
    )

# 새로운 체크아웃 라우트
@senior.route('/checkout')
def checkout(): 
    session.pop("cart", None)  # 카트 비우기
    return redirect(url_for('senior.home'))  # 체크아웃 라우트
  # 메인 페이지로 리디렉션


# 공통 라우트 정의
@app.route('/set_language/<string:lang>', methods=['POST'])
@senior.route('/set_language/<string:lang>', methods=['POST'])
def set_language(lang):
    # Flask 세션에 언어 저장
    session['language'] = lang
    return jsonify(success=True, language=lang)

@app.route('/get_language', methods=['GET'])
@senior.route('/get_language', methods=['GET'])
def get_language():
    current_language = session.get('language', 'ko')  # 기본값: 'ko'
    return jsonify(language=current_language)
# Blueprint 등록
app.register_blueprint(senior)

if __name__ == "__main__":
    app.run(debug=True)