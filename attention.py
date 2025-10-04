import cv2
import mediapipe as mp
import numpy as np
import base64

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


def eye_centered(eye_left, eye_right, iris):
    eye_width = eye_right.x - eye_left.x
    iris_offset = (iris.x - eye_left.x) / eye_width
    return 0.35 < iris_offset < 0.65


def process_batch(frames_b64):
    results_list = []

    for frame_b64 in frames_b64:
        # Decode base64 → numpy array
        img_data = base64.b64decode(frame_b64)
        np_arr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        attentive_status = "Not Attentive"

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                chin = face_landmarks.landmark[CHIN]
                forehead = face_landmarks.landmark[FOREHEAD]
                nose = face_landmarks.landmark[NOSE]
                left_eye = face_landmarks.landmark[LEFT_EYE_LEFT]
                right_eye = face_landmarks.landmark[RIGHT_EYE_RIGHT]
                left_iris = face_landmarks.landmark[LEFT_IRIS]
                right_iris = face_landmarks.landmark[RIGHT_IRIS]

                dist_left = abs(nose.x - left_eye.x)
                dist_right = abs(right_eye.x - nose.x)
                looking_sideways = abs(dist_left - dist_right) > 0.03

                depth_diff = chin.z - forehead.z
                looking_down = depth_diff < -0.1
                looking_up = depth_diff > 0.1

                left_ok = eye_centered(left_eye, face_landmarks.landmark[LEFT_EYE_RIGHT], left_iris)
                right_ok = eye_centered(face_landmarks.landmark[RIGHT_EYE_LEFT], right_eye, right_iris)
                # looking_at_screen = left_ok and right_ok
                looking_at_screen = True  # simplifying

                if not looking_sideways and not looking_down and not looking_up and looking_at_screen:
                    attentive_status = "Attentive"

        results_list.append(1 if attentive_status == "Attentive" else 0)

    # Probability = ratio of attentive frames in batch
    probability = sum(results_list) / len(results_list) if results_list else 0.0
    batch_result = "Attentive" if probability >= 0.5 else "Not Attentive"

    return {
        "batchResult": batch_result,
        "probability": round(probability, 2)
    }