import cv2
import mediapipe as mp
import math

cap = cv2.VideoCapture(0)

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

count = 0
direction = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)

    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

        lmList = []
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, _ = img.shape
            lmList.append((id, int(lm.x * w), int(lm.y * h)))

        if len(lmList) != 0:
            x1, y1 = lmList[11][1:]
            x2, y2 = lmList[13][1:]
            x3, y3 = lmList[15][1:]

            angle = math.degrees(
                math.atan2(y3 - y2, x3 - x2) -
                math.atan2(y1 - y2, x1 - x2)
            )

            if angle < 0:
                angle += 360

            per = int((angle - 30) * 100 / (160 - 30))

            if per == 100:
                if direction == 0:
                    count += 0.5
                    direction = 1

            if per == 0:
                if direction == 1:
                    count += 0.5
                    direction = 0

            cv2.putText(img, f'Reps: {int(count)}', (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 3)

    cv2.imshow("Gym Tracker", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
