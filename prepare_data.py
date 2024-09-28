import os
import cv2
import numpy as np
import random
import mediapipe as mp
from utils import get_face_landmarks, IMPORTANT_LANDMARKS

# Đặt đường dẫn thư mục cảm xúc (ví dụ: './data/happy')
data_dir = './data/neutral'

# Đặt tên file kết quả tương ứng với cảm xúc (happy_data.txt, neutral_data.txt, ...)
output_file = 'neutral_data.txt'

output = []

# Lấy tất cả các ảnh trong thư mục cảm xúc
image_paths = os.listdir(data_dir)
num_images = len(image_paths)

# Bạn có thể điều chỉnh cách chọn ảnh (ở đây đang lấy tất cả các ảnh)
selected_images = random.sample(image_paths, num_images)

# Khởi tạo Mediapipe FaceMesh một lần
mp_face_mesh = mp.solutions.face_mesh

with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
) as face_mesh:

    for image_name in selected_images:
        image_path = os.path.join(data_dir, image_name)

        image = cv2.imread(image_path)

        # Lấy landmarks từ ảnh
        face_landmarks = get_face_landmarks(image, face_mesh)

        # Kiểm tra xem landmarks có hợp lệ không
        expected_landmark_length = len(IMPORTANT_LANDMARKS) * 3
        if face_landmarks and len(face_landmarks) == expected_landmark_length:
            # Gán nhãn cho cảm xúc hiện tại (ví dụ: 0 cho happy)
            emotion_index = 0  # Thay đổi nhãn phù hợp với cảm xúc
            face_landmarks.append(emotion_index)  # Thêm label vào cuối landmarks
            output.append(face_landmarks)
            print(f"Số lượng điểm landmarks: {len(face_landmarks) // 3}")  # In ra số lượng điểm
        else:
            print(f"Lỗi khi xử lý ảnh: {image_name}")

# Sau khi hoàn thành, lưu dữ liệu vào file txt tương ứng với cảm xúc
np.savetxt(output_file, np.asarray(output), fmt='%f')

print(f"Lưu dữ liệu của cảm xúc vào {output_file} thành công!")
