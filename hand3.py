import cv2
import mediapipe as mp
import math

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Function to check which fingers are up
def fingers_up(hand_landmarks, hand_label):
    finger_tips = [8, 12, 16, 20]  # index, middle, ring, pinky
    finger_pips = [6, 10, 14, 18]

    fingers = []

    # Thumb check depends on left/right
    if hand_label == "Right":
        fingers.append(1 if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x else 0)
    else:  # Left hand
        fingers.append(1 if hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x else 0)

    # Other fingers
    for tip, pip in zip(finger_tips, finger_pips):
        fingers.append(1 if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y else 0)

    return fingers

# Map finger patterns to single-hand gestures
def detect_gesture(fingers):
    if fingers == [0,0,0,0,0]:
        return "Fist"
    elif fingers == [1,0,0,0,0]:
        return "Thumbs Up"
    elif fingers == [0,1,1,0,0]:
        return "Peace"
    elif fingers == [1,1,1,1,1]:
        return "Open Palm"
    elif fingers == [1,1,0,0,1]:
        return "YO"
    else:
        return "Other"

# Function to calculate distance between two landmarks
def distance(lm1, lm2):
    return math.hypot(lm1.x - lm2.x, lm1.y - lm2.y)

# Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    gesture_text = "No Hand"
    hand_gestures = {}
    hand_centers = {}

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label  # "Left" or "Right"
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            fingers = fingers_up(hand_landmarks, label)
            gesture = detect_gesture(fingers)

            # use palm center (landmark[9]) as reference
            hand_centers[label] = hand_landmarks.landmark[9]
            hand_gestures[label] = gesture

        # Two-hand gesture check
        if "Left" in hand_gestures and "Right" in hand_gestures:
            left = hand_gestures["Left"]
            right = hand_gestures["Right"]

            # distance between palm centers
            d = distance(hand_centers["Left"], hand_centers["Right"])

            if left == "Open Palm" and right == "Open Palm":
                if d < 0.01:   # hands very close
                    gesture_text = "Namaste "
                else:
                    gesture_text = "Clapping "
            elif left == "Thumbs Up" and right == "Thumbs Up":
                gesture_text = "Double Thumbs Up"
            elif left == "Peace" and right == "Peace":
                gesture_text = "Double Peace"
            elif (left == "Open Palm" and right == "Fist") or (left == "Fist" and right == "Open Palm"):
                gesture_text = "Handshake"
            elif left == "Fist" and right == "Fist":
                gesture_text = "Fist bump"
            else:
                gesture_text = f"L:{left}, R:{right}"
        else:
            # Only one hand detected
            gesture_text = list(hand_gestures.values())[0]

    cv2.putText(frame, gesture_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 0), 2)

    cv2.imshow("Single + Both Hand Gestures", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
