from imutils import face_utils
import cv2
import dlib
import imutils
import time


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('path to shape_predictor_68_face_landmarks.dat')
time_to_warmup_camera = 2
debug = False
(lStart,lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
cap = cv2.VideoCapture(0)
time.sleep(time_to_warmup_camera)

def slope_of_line(pt1, pt2):
    return (round(abs(pt2[1] - pt1[1]) / abs(pt2[0] - pt1[0]), 2))


while True:
    _, frame = cap.read()
    frame = imutils.resize(frame,width=500)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    rects = detector(gray,0)
    for rect in rects:
        try:
            shape = predictor(gray,rect)
            shape = face_utils.shape_to_np(shape)
            lips = shape[lStart:lEnd]
            y_left = lips[0][1]
            y_mid = lips[3][1]
            y_right = lips[6][1]
            slope1 = slope_of_line(lips[6], lips[3])
            slope2 = slope_of_line(lips[3], lips[0])
            if slope1 < 0.15 and slope2 < 0.15:
                cv2.putText(frame, "Happy", (250,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
            else:
                cv2.putText(frame, "Sad", (250,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
            if debug:
                lipsHull = cv2.convexHull(lips)
                cv2.putText(frame,"Dst: {} {} {} {} {} {}".format(slope1, slope2, y_right, y_mid - y_left, y_right - y_mid, y_right - y_left),(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

        except Exception as e:
            print(str(e))
            pass        
    cv2.imshow('face',frame)
    if cv2.waitKey(1) & 0xff == 27:
        break

cap.release()
cv2.destroyAllWindows()