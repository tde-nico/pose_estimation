import cv2
import mediapipe as mp
import time
import Pose_Estimation_Module as pem


cap = cv2.VideoCapture(0) # 1 ?
pTime = 0
cTime = 0
detector = pem.PoseDetector()
while 1:
	success, img = cap.read()
	img = detector.find_pose(img)
	lm_list = detector.find_position(img)
	#print(lm_list)

	cTime = time.time()
	fps = 1/(cTime - pTime)
	pTime = cTime

	cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
	cv2.imshow("Image", img)
	cv2.waitKey(1)