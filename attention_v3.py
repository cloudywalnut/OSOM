# Some fine tuning needed but way better from prev versions
import cv2
import mediapipe as mp

# Initialize mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

# Landmark indices
CHIN = 152
FOREHEAD = 10
NOSE = 1
RIGHT_EYE_RIGHT = 263
LEFT_EYE_LEFT = 33
LEFT_IRIS = 468
RIGHT_IRIS = 473
LEFT_EYE_RIGHT = 133
RIGHT_EYE_LEFT = 362


# Start webcam
cap = cv2.VideoCapture(0)

# Gaze detection
def eye_centered(eye_left, eye_right, iris):
    eye_width = eye_right.x - eye_left.x
    iris_offset = (iris.x - eye_left.x) / eye_width
    return 0.35 < iris_offset < 0.65


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    attentive_status = "Not Present"

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            # Getting the important points
            chin = face_landmarks.landmark[CHIN]
            forehead = face_landmarks.landmark[FOREHEAD]
            nose = face_landmarks.landmark[NOSE]
            left_eye = face_landmarks.landmark[LEFT_EYE_LEFT]
            right_eye = face_landmarks.landmark[RIGHT_EYE_RIGHT]
            left_iris = face_landmarks.landmark[LEFT_IRIS]
            right_iris = face_landmarks.landmark[RIGHT_IRIS]


            # Head rotation detection using eye-nose distances
            dist_left = abs(nose.x - left_eye.x)
            dist_right = abs(right_eye.x - nose.x)
            looking_sideways = abs(dist_left - dist_right) > 0.03  # threshold to tune

            # Check up/down using z difference
            depth_diff = chin.z - forehead.z
            looking_down = depth_diff < -0.1   # chin closer
            looking_up = depth_diff > 0.1      # forehead closer

            # Gaze detection if eyes are looking somewhere else
            left_ok = eye_centered(left_eye, face_landmarks.landmark[LEFT_EYE_RIGHT], left_iris)
            right_ok = eye_centered(face_landmarks.landmark[RIGHT_EYE_LEFT], right_eye, right_iris)
            # looking_at_screen = left_ok and right_ok
            looking_at_screen = True # Setting true for now else might make the rules too rigid or strict

            # Condition to determine if attentive or not
            if not looking_sideways and not looking_down and not looking_up and looking_at_screen:
                attentive_status = "Attentive"
            else:
                attentive_status = "Not Attentive"

            # Draw landmarks (optional)
            mp_drawing.draw_landmarks(
                frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1)
            )

            cv2.putText(frame, attentive_status, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255) if attentive_status != "Attentive" else (0, 255, 0), 2)

    else:
        cv2.putText(frame, "Not Present", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 2)

    cv2.imshow("Attention Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
