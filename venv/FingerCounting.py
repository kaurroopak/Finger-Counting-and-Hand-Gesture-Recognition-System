import cv2
import time
import os

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

folderPath = r"C:\Users\DELL\Finger-Counting-System\FingerImages"
myList = os.listdir(folderPath)
print(myList)
overlayList = []
for iPath in myList:
    image = cv2.imread(f'{folderPath}/{iPath}')
    # print(f'{folderPath}/{iPath}')
    overlayList.append(image)
print(len(overlayList))

previousTime = 0
while True:
    success, img = cap.read()

    overlay_small = cv2.resize(overlayList[0], (180, 180))
    img[0:180, 0:180] = overlay_small

    currentTime = time.time()
    fps = 1/(currentTime - previousTime)
    previousTime = currentTime

    cv2.putText(img, f'FPS: {int(fps)}', (400,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)