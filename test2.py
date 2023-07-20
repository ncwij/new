import cv2
import mediapipe as mp
import time


class poseDetector():

    def __init__(self, mode = False, model_complexity = 1, smooth = True,
                 enable_segmentation = False,smooth_segmentation = True, detectionCon=0.5, trackCon=0.5 ):

        self.mode = mode
        self.model_complexity = model_complexity
        self.smooth = smooth
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.model_complexity,self.smooth,
                                     self.enable_segmentation, self.smooth_segmentation, self.detectionCon, self.trackCon)

    def findPose(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
        return img


    def findPosition(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                #print(id, lm)
                cx, cy =int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx,cy])
                if draw:
                    cv2.circle(img, (cx,cy), 10, (255, 0, 0), cv2.FILLED)
        return lmList




def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = poseDetector()
    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.findPosition(img,draw = False)
        if len(lmList) !=0:
            print(lmList[11], lmList[12], lmList[23], lmList[24])
            cv2.circle(img, (lmList[11][1], lmList[11][2]), 20, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (lmList[12][1], lmList[11][2]), 20, (0, 255, 0), cv2.FILLED)
            cv2.circle(img, (lmList[23][1], lmList[23][2]), 20, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (lmList[24][1], lmList[24][2]), 20, (255, 255, 0), cv2.FILLED)
            center = (lmList[11][1]+lmList[12][1])/2
            print(center)
            if 250<=center<=360:
                print('good')
            else:
                print('bad')
        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 0),3)

        cv2.imshow('Image', img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()