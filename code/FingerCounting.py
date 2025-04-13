import cv2
import time
import os
import HandTrackingModule as htm

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# Use relative path instead of hardcoded absolute path
folderPath = os.path.join(os.getcwd(), 'FingerImages')

# Check if the folder exists
if not os.path.exists(folderPath):
    raise FileNotFoundError(f"Folder 'FingerImages' not found at: {folderPath}")

myList = os.listdir(folderPath)
print(myList)

overlayList = []
for iPath in myList:
    image = cv2.imread(os.path.join(folderPath, iPath))
    overlayList.append(image)

print(f"{len(overlayList)} images loaded.")

previousTime = 0

detector = htm.handDetector(detectionCon=0.75)
tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        fingers = []
        # Thumb
        if lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # 4 Fingers (Index to Pinky)
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        totalFingers = fingers.count(1)
        print(totalFingers)

        if totalFingers > 0 and totalFingers <= len(overlayList):
            overlay_small = cv2.resize(overlayList[totalFingers - 1], (180, 180))
            img[0:180, 0:180] = overlay_small

        cv2.rectangle(img, (20, 225), (170, 425), (255, 255, 255), cv2.FILLED)
        cv2.putText(img, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN, 10, (0, 0, 0), 25)

    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime

    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
