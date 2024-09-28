import cv2
import mediapipe as mp
from utils import get_face_landmarks

# Khởi tạo Mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh

# Khởi tạo FaceMesh một lần duy nhất để tránh khởi tạo lại mỗi lần xử lý
with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    # Đọc từ webcam
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Không thể lấy được khung hình.")
            break

        # Lấy các điểm landmarks quan trọng từ hàm get_face_landmarks
        important_landmarks = get_face_landmarks(image, face_mesh)

        # Kiểm tra nếu lấy được các điểm landmarks
        if important_landmarks:
            print("Các điểm landmarks quan trọng:", important_landmarks)

            # Vẽ các điểm landmarks lên ảnh
            h_image, w_image, _ = image.shape
            for i in range(0, len(important_landmarks), 3):
                # Tọa độ các điểm
                landmark_x = int(important_landmarks[i] * w_image)
                landmark_y = int(important_landmarks[i + 1] * h_image)

                # Vẽ chấm tròn tại vị trí landmark
                cv2.circle(image, (landmark_x, landmark_y), 2, (0, 0, 255), -1)

        # Hiển thị hình ảnh
        cv2.imshow('Face Landmarks', image)

        if cv2.waitKey(5) & 0xFF == 27:  # Nhấn ESC để thoát
            break

    # Dừng webcam và giải phóng bộ nhớ khi kết thúc
    cap.release()
    cv2.destroyAllWindows()
