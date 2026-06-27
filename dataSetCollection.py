import cv2
import mediapipe as mp
import os

# Dataset Settings

LABEL = "A"                 # Change for each gesture
IMG_SIZE = 224              # Final image size
OFFSET = 40                 # Padding around hand

SAVE_PATH = os.path.join("dataset", LABEL)
os.makedirs(SAVE_PATH, exist_ok=True)

count = len(os.listdir(SAVE_PATH))

# MediaPipe Initialization

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# Camera

cap = cv2.VideoCapture(0)

print("\n==============================")
print("Press 'S' to save image")
print("Press 'Q' to quit")
print("==============================\n")

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    # Copy of original frame (NO drawings)
    clean_frame = frame.copy()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        for hand in results.multi_hand_landmarks:

            
            # Draw only for display
           
            mp_draw.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS
            )

            h, w, _ = frame.shape

            x_list = []
            y_list = []

            for lm in hand.landmark:
                x = int(lm.x * w)
                y = int(lm.y * h)

                x_list.append(x)
                y_list.append(y)

            x_min = min(x_list)
            x_max = max(x_list)
            y_min = min(y_list)
            y_max = max(y_list)

            # Add padding
            x_min -= OFFSET
            y_min -= OFFSET
            x_max += OFFSET
            y_max += OFFSET

            # Keep inside image
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(w, x_max)
            y_max = min(h, y_max)

            # Warning if hand is touching border
            if x_min == 0 or y_min == 0 or x_max == w or y_max == h:

                cv2.putText(
                    frame,
                    "Move Hand Inside Frame",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

            # Draw Bounding Box (Preview Only)
            cv2.rectangle(
                frame,
                (x_min, y_min),
                (x_max, y_max),
                (0, 255, 0),
                2
            )

            
            # Square Crop
           

            width = x_max - x_min
            height = y_max - y_min

            side = max(width, height)

            center_x = (x_min + x_max) // 2
            center_y = (y_min + y_max) // 2

            new_x_min = max(0, center_x - side // 2)
            new_y_min = max(0, center_y - side // 2)

            new_x_max = min(w, new_x_min + side)
            new_y_max = min(h, new_y_min + side)

            # Crop from CLEAN frame
            hand_crop = clean_frame[
                new_y_min:new_y_max,
                new_x_min:new_x_max
            ]

            if hand_crop.size != 0:

                hand_crop = cv2.resize(
                    hand_crop,
                    (IMG_SIZE, IMG_SIZE)
                )

                cv2.imshow("Hand Crop", hand_crop)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('s'):

                    filename = os.path.join(
                        SAVE_PATH,
                        f"{count}.jpg"
                    )

                    cv2.imwrite(filename, hand_crop)

                    count += 1

                    print(f"Image {count} saved.")

                elif key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()

    cv2.putText(
        frame,
        f"Images : {count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )

    cv2.imshow("Dataset Collector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()