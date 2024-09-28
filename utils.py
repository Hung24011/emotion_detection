import cv2
import mediapipe as mp

# Lông mày
LANDMARKS_EYEBROWS = [70, 63, 105, 66, 107, 336, 296, 334]

# Mắt
LANDMARKS_EYES = [33, 160, 158, 133, 153, 144, 362, 387, 385, 263, 373, 380]

# Miệng
LANDMARKS_MOUTH = [78, 95, 88, 178, 87, 14, 317, 402, 310, 415, 308, 324]

# Mũi và má
LANDMARKS_NOSE_CHEEKS = [1, 2, 98, 327, 94, 331, 78, 82, 13, 312]

# Viền khuôn mặt (Face Contour)
LANDMARKS_FACE_CONTOUR = [
    10, 338, 297, 332, 284, 251, 389, 356, 454,
    323, 361, 288, 397, 365, 379, 378, 400, 377,
    152, 148, 176, 149, 150, 136, 172, 58, 132,
    93, 234, 127, 162, 21, 54, 103, 67, 109
]

# Kết hợp tất cả các điểm landmarks quan trọng
IMPORTANT_LANDMARKS = (
    LANDMARKS_EYEBROWS + LANDMARKS_EYES + LANDMARKS_MOUTH +
    LANDMARKS_NOSE_CHEEKS + LANDMARKS_FACE_CONTOUR
)

def get_face_landmarks(image, face_mesh, draw=False):
    """
    Hàm để lấy các điểm landmarks quan trọng của khuôn mặt từ ảnh.
    """
    # Chuyển đổi ảnh từ BGR sang RGB
    image_input_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Xử lý ảnh để tìm khuôn mặt
    results = face_mesh.process(image_input_rgb)

    image_landmarks = []

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0].landmark

        if draw:
            mp_drawing = mp.solutions.drawing_utils
            drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=results.multi_face_landmarks[0],
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)

        xs_ = []
        ys_ = []
        zs_ = []

        # Lấy tọa độ của các điểm mốc quan trọng
        for idx in IMPORTANT_LANDMARKS:
            lm = face_landmarks[idx]
            xs_.append(lm.x)
            ys_.append(lm.y)
            zs_.append(lm.z)

        # Chuẩn hóa các điểm landmarks
        min_x, min_y, min_z = min(xs_), min(ys_), min(zs_)
        for j in range(len(xs_)):
            image_landmarks.append(xs_[j] - min_x)
            image_landmarks.append(ys_[j] - min_y)
            image_landmarks.append(zs_[j] - min_z)
    else:
        print("Không tìm thấy khuôn mặt.")

    return image_landmarks
