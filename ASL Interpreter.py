import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
mp_Hands = mp.solutions.hands
hands = mp_Hands.Hands()
mpDraw = mp.solutions.drawing_utils

while True:
    success, image = cap.read()
    image = cv2.flip(image, 1)
    RGB_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(RGB_image)
    multiLandMarks = results.multi_hand_landmarks

    if multiLandMarks:
        handList = []
        detected_letter = None

        for handLms in multiLandMarks:
            mpDraw.draw_landmarks(image, handLms, mp_Hands.HAND_CONNECTIONS)
            for idx, lm in enumerate(handLms.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                handList.append((cx, cy))
                cv2.circle(image, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

        # Debugging output: Print detected landmarks
        #print("Detected Landmarks:")
        #for i, point in enumerate(handList):
        #    print(f"Landmark {i}: {point}")


        if all(handList[4][1] < handList[i][1] for i in range(0, 21) if i != 4):
            detected_letter = 'A'

        if (
                handList[12][0] > handList[8][0] and
                handList[16][0] > handList[8][0] and
                handList[20][0] > handList[4][0]
        ):
            detected_letter = 'B'

        if (
                handList[8][0] < handList[6][0] and
                handList[12][0] < handList[10][0] and
                handList[16][0] < handList[14][0] and
                handList[20][0] < handList[18][0]
        ):
            detected_letter = 'C'

        if all(handList[8][1] < handList[i][1] for i in range(0, 21) if i != 8):
            detected_letter = 'D'


        if all(handList[i][1] < handList[j][1] for i, j in zip([6, 10, 14, 18], [8, 12, 16, 20])):
            detected_letter = 'A'


            if handList[4][1] > min(handList[i][1] for i in range(0, 21) if i not in [4]):
                detected_letter = 'E'

        if detected_letter:
            cv2.putText(image, f'Detected Letters: {detected_letter}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)


    flipped_image = cv2.flip(image, 1)

    cv2.imshow("Sign Language Interpreter", image)
    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
