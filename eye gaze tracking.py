import cv2
import dlib
import numpy as np

# Load dlib's face detector and facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Ensure this file is present

# 3D model points for head pose estimation (reference face landmarks)
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),  # Nose tip
    (0.0, -330.0, -65.0),  # Chin
    (-225.0, 170.0, -135.0),  # Left eye corner
    (225.0, 170.0, -135.0),  # Right eye corner
    (-150.0, -150.0, -125.0),  # Left mouth corner
    (150.0, -150.0, -125.0)  # Right mouth corner
])

# Camera matrix (assuming a 640x480 webcam)
FOCAL_LENGTH = 640
CAMERA_MATRIX = np.array([
    [FOCAL_LENGTH, 0, 320],
    [0, FOCAL_LENGTH, 240],
    [0, 0, 1]
], dtype="double")

# Indices for facial landmarks
LEFT_EYE_POINTS = list(range(36, 42))
RIGHT_EYE_POINTS = list(range(42, 48))

def get_head_pose(landmarks):
    """Estimate head pose using facial landmarks."""
    image_points = np.array([
        (landmarks.part(30).x, landmarks.part(30).y),  # Nose tip
        (landmarks.part(8).x, landmarks.part(8).y),  # Chin
        (landmarks.part(36).x, landmarks.part(36).y),  # Left eye corner
        (landmarks.part(45).x, landmarks.part(45).y),  # Right eye corner
        (landmarks.part(48).x, landmarks.part(48).y),  # Left mouth corner
        (landmarks.part(54).x, landmarks.part(54).y)   # Right mouth corner
    ], dtype="double")

    # SolvePnP to estimate rotation
    _, rotation_vector, _ = cv2.solvePnP(MODEL_POINTS, image_points, CAMERA_MATRIX, None)
    return rotation_vector

def get_eye_position(landmarks):
    """Determine if the eyes are closed or detect gaze direction."""
    left_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in LEFT_EYE_POINTS]
    right_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in RIGHT_EYE_POINTS]

    # Compute eye aspect ratio to check for eye closure
    left_eye_aspect_ratio = eye_aspect_ratio(left_eye)
    right_eye_aspect_ratio = eye_aspect_ratio(right_eye)

    # If the aspect ratio is below a threshold, consider the eye closed
    EAR_THRESHOLD = 0.2
    if left_eye_aspect_ratio < EAR_THRESHOLD and right_eye_aspect_ratio < EAR_THRESHOLD:
        return "Eyes Closed"

    # Compute eye center for gaze direction
    left_center = np.mean(left_eye, axis=0)
    right_center = np.mean(right_eye, axis=0)

    # Compare pupil position with eye width
    eye_width = left_eye[-1][0] - left_eye[0][0]
    left_ratio = (left_center[0] - left_eye[0][0]) / eye_width
    right_ratio = (right_center[0] - right_eye[0][0]) / eye_width

    if left_ratio < 0.35 and right_ratio < 0.35:
        return "Looking Left"
    elif left_ratio > 0.65 and right_ratio > 0.65:
        return "Looking Right"
    else:
        return "Looking Center"

def eye_aspect_ratio(eye):
    """Compute the aspect ratio of the eye to determine if it is closed."""
    vertical_dist1 = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
    vertical_dist2 = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
    horizontal_dist = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
    ear = (vertical_dist1 + vertical_dist2) / (2.0 * horizontal_dist)
    return ear

def determine_gaze(landmarks):
    """Combine head pose and eye position to determine exact gaze direction."""
    rotation_vector = get_head_pose(landmarks)
    eye_direction = get_eye_position(landmarks)

    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    # Decompose rotation matrix to Euler angles
    sy = np.sqrt(rotation_matrix[0, 0] * rotation_matrix[0, 0] + rotation_matrix[1, 0] * rotation_matrix[1, 0])
    singular = sy < 1e-6

    if not singular:
        pitch = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        yaw = np.arctan2(-rotation_matrix[2, 0], sy)
        roll = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        pitch = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        yaw = np.arctan2(-rotation_matrix[2, 0], sy)
        roll = 0

    # Convert from radians to degrees
    pitch, yaw, roll = np.degrees([pitch, yaw, roll])

    # Analyze head tilt (pitch) and rotation (yaw)
    if yaw < -10:
        head_direction = "Head Turned Left"
    elif yaw > 10:
        head_direction = "Head Turned Right"
    elif pitch < -10:
        head_direction = "Looking Up"
    elif pitch > 10:
        head_direction = "Looking Down"
    else:
        head_direction = "Head Centered"

    # Final gaze determination
    if "Left" in head_direction or "Left" in eye_direction:
        return "Looking Left"
    elif "Right" in head_direction or "Right" in eye_direction:
        return "Looking Right"
    elif "Up" in head_direction:
        return "Looking Up"
    elif "Down" in head_direction:
        return "Looking Down"
    elif "Eyes Closed" in eye_direction:
        return "Eyes Closed"
    else:
        return "Looking Center"

def apply_semi_transparent_mask(frame, landmarks):
    """Apply a semi-transparent mask on the face."""
    mask = np.zeros_like(frame, dtype=np.uint8)

    # Get facial landmarks (68 points)
    points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)], dtype=np.int32)
    hull = cv2.convexHull(points)  # Convex hull around face

    # Draw semi-transparent mask on face area
    cv2.fillConvexPoly(mask, hull, (0, 0, 255))  # Red mask for visibility

    # Blend the mask with the original frame
    alpha = 0.3  # Mask transparency
    frame = cv2.addWeighted(frame, 1 - alpha, mask, alpha, 0)

    return frame

def apply_graphic_mask(frame, landmarks, mask_image_path="mask.png"):
    """Apply a graphic mask image over the face."""
    # Load the custom mask image (ensure it's a PNG with transparency)
    mask_img = cv2.imread(mask_image_path, cv2.IMREAD_UNCHANGED)

    # Get face region (bounding box)
    x, y, w, h = cv2.boundingRect(np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)], dtype=np.int32))
    mask_resized = cv2.resize(mask_img, (w, h))

    # Ensure the mask has an alpha channel
    if mask_resized.shape[2] == 4:
        # Separate channels (BGR + Alpha)
        alpha_channel = mask_resized[:, :, 3] / 255.0
        bgr_mask = mask_resized[:, :, :3]
        for c in range(3):
            frame[y:y+h, x:x+w, c] = frame[y:y+h, x:x+w, c] * (1 - alpha_channel) + bgr_mask[:, :, c] * alpha_channel

    return frame

# Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) == 0:
        print("No faces detected")  # Debug message if no faces are detected

    for face in faces:
        landmarks = predictor(gray, face)
        gaze = determine_gaze(landmarks)

        # Display gaze direction text
        cv2.putText(frame, gaze, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Apply semi-transparent or graphic mask
        frame = apply_semi_transparent_mask(frame, landmarks)  # Use this line for semi-transparent mask
        # frame = apply_graphic_mask(frame, landmarks)  # Use this line for graphic mask

    cv2.imshow("Full Face & Eye Gaze Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()