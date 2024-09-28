import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model
from utils import get_face_landmarks, IMPORTANT_LANDMARKS
import joblib  # Import joblib để load scaler đã lưu

# Load mô hình phân loại cảm xúc
model = load_model('model_data_12.keras')

# Load scaler đã lưu
scaler = joblib.load('scaler.pkl')

# Khởi tạo Mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh

# Khởi tạo Cascade Classifier để phát hiện khuôn mặt
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Danh sách nhãn cảm xúc
emotion_labels = ['Anger', 'Happy', 'Neutral', 'Sadness']

def preprocess_for_emotion_classification(landmarks):
    """
    Chuẩn bị dữ liệu để đưa vào mô hình phân loại cảm xúc.
    """
    landmark_array = np.array(landmarks)
    
    expected_input_length = len(IMPORTANT_LANDMARKS) * 3  # Số lượng điểm * 3 tọa độ (x, y, z)
    if len(landmark_array) < expected_input_length:
        landmark_array = np.pad(landmark_array, (0, expected_input_length - len(landmark_array)), 'constant')
    
    # Chuẩn hóa dữ liệu bằng scaler
    landmark_array = landmark_array.reshape(1, -1)  # Đảm bảo reshape thành vector 1D trước khi chuẩn hóa
    return scaler.transform(landmark_array)

# Khởi tạo FaceMesh và video capture
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    static_image_mode=False,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:
    
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Không thể lấy được khung hình.")
            break

        # Chuyển đổi ảnh sang grayscale để dùng với Haar Cascade
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Phát hiện khuôn mặt bằng Haar Cascade
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            # Vẽ khung hình chữ nhật quanh khuôn mặt
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Cắt phần khuôn mặt được phát hiện
            face_roi = frame[y:y+h, x:x+w]

            # Lấy landmarks từ ảnh khuôn mặt
            landmarks = get_face_landmarks(face_roi, face_mesh)

            if landmarks and len(landmarks) == len(IMPORTANT_LANDMARKS) * 3:
                h_roi, w_roi, _ = face_roi.shape

                # Vẽ các điểm landmarks lên ảnh đã cắt
                for i in range(0, len(landmarks), 3):
                    landmark_x = int(landmarks[i] * w_roi)
                    landmark_y = int(landmarks[i + 1] * h_roi)
                    cv2.circle(face_roi, (landmark_x, landmark_y), 2, (0, 0, 255), -1)

                # Chuẩn bị dữ liệu để phân loại cảm xúc
                input_data = preprocess_for_emotion_classification(landmarks)

                # Dự đoán cảm xúc
                emotion_prediction = model.predict(input_data)
                predicted_emotion = np.argmax(emotion_prediction)
                predicted_emotion_label = emotion_labels[predicted_emotion]

                # Hiển thị kết quả phân loại cảm xúc trên ảnh đã cắt
                cv2.putText(frame, f'Emotion: {predicted_emotion_label}', (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (36, 255, 12), 2)

        # Hiển thị ảnh đã xử lý
        cv2.imshow('Face Landmarks and Emotion Detection', frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
