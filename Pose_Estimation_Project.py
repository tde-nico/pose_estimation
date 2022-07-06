import cv2
import mediapipe as mp
import time


mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0) # 1 ?

pTime = 0
cTime = 0

while 1:
	success, img = cap.read()
	imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	results = pose.process(imgRGB)

	lms = results.pose_landmarks
	if lms:
		for id, lm in enumerate(lms.landmark):
			#print(id, lm)
			h, w, c = img.shape
			cx, cy = int(lm.x * w), int(lm.y * h)
			#print(id, cx, cy)
			if id == 4:
				cv2.circle(img, (cx,cy), 5, (255,0,255), cv2.FILLED)
		mpDraw.draw_landmarks(img, lms, mpPose.POSE_CONNECTIONS)

	cTime = time.time()
	fps = 1/(cTime - pTime)
	pTime = cTime

	cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
	cv2.imshow("Image", img)
	cv2.waitKey(1)

