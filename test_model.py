import  cv2
from keras.models import load_model
from utils import get_face_landmarks, IMPORTANT_LANDMARKS


model = load_model('model_data_12.keras')


cap = cv2.VideoCapture(0)

ret , frame = cap.read()

while ret:
    ret, frame = cap.read()
    
    face_landmarks = get_face_landmarks(frame, face_mesh, draw=True)
    
    
    cv2.imshow('frame', frame)
    cv2.waitKey(25)
    
cap.release()
cv2.destroyAllWindows()