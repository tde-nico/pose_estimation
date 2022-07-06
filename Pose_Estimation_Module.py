import cv2
import mediapipe as mp
import time


class PoseDetector:
	def __init__(self, mode=False, model_complexity=1, upper_body=False, smooth=True, detection_con=0.5, track_con=0.5):
		self.mode = mode
		self.model_complexity = model_complexity
		self.upper_body = upper_body
		self.smooth = smooth
		self.detection_con = detection_con
		self.track_con = track_con
		self.mpPose = mp.solutions.pose
		self.pose = self.mpPose.Pose(mode, model_complexity, upper_body, smooth, detection_con, track_con)
		self.mpDraw = mp.solutions.drawing_utils


	def find_pose(self, img, draw=True):
		imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		results = self.pose.process(imgRGB)
		self.lms = results.pose_landmarks
		if self.lms and draw:
				self.mpDraw.draw_landmarks(img, self.lms, self.mpPose.POSE_CONNECTIONS)
		return img


	def find_position(self, img, draw=True):
		lm_list = []
		for id, lm in enumerate(self.lms.landmark):
			#print(id, lm)
			h, w, c = img.shape
			cx, cy = int(lm.x * w), int(lm.y * h)
			#print(id, cx, cy)
			lm_list.append([id, cx, cy])
			if draw:
				cv2.circle(img, (cx,cy), 10, (255,0,255), cv2.FILLED)
		return lm_list


def main():
	cap = cv2.VideoCapture(0) # 1 ?
	pTime = 0
	cTime = 0
	detector = PoseDetector()
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

if __name__ == '__main__':
	main()
