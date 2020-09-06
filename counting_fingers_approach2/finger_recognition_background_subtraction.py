import cv2, numpy as np
from sklearn.metrics import pairwise
import imutils, time


'''This method is used to separate foreground from background. In this concept, the video sequence is analyzed over a particular set of frames.
   During this sequence of frames, the running average over the current frame and the previous frames is computed'''
def cal_running_avg(frame, accumulated_weight):
    global background
    if background is None:
        background = frame.copy().astype('float')
        return None
    cv2.accumulateWeighted(frame, background, accumulated_weight)

	
'''This method is used to subtract background and frame to get the hand and find the maximum contour on it.
   thresh_min value can be changed and experimented with as it may vary according to your lighting conditions'''
def segment(frame, thresh_min=50):
    diff = cv2.absdiff(background.astype('uint8'), frame)
    _, thresh = cv2.threshold(diff, thresh_min, 255, cv2.THRESH_BINARY)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    if len(cnts) > 0:
        hand_segment = max(cnts, key=cv2.contourArea)
        return (thresh, hand_segment)
    else:
        return None
		

'''This method finds the convex hull and the center of the convex hull.
   Then it finds the pair wise distance between center and top, left, bottom and right of convex hull
   It finds a circle of assumed 80% radius of the max pairwise distance
   The contours formed outside the circle are fingers'''
def get_fingers_count(thresh_frame, hand_segment):
    conv_hull = cv2.convexHull(hand_segment)
    top = tuple(conv_hull[conv_hull[:,:,1].argmin()][0])
    bottom = tuple(conv_hull[conv_hull[:,:,1].argmax()][0])
    left = tuple(conv_hull[conv_hull[:,:,1].argmin()][0])
    right = tuple(conv_hull[conv_hull[:,:,1].argmax()][0])
    
    cX = (left[0] + right[0]) // 2
    cY = (top[1] + bottom[1]) // 2
    
    distance = pairwise.euclidean_distances([(cX,cY)], Y=[left,right,top,bottom])[0]
    
    max_dist = distance.max()
    if max_dist:
        radius = int(0.8 * max_dist)
        circumference = (2 * np.pi * radius)
        circular_roi = np.zeros(thresh_frame.shape[:2], dtype=np.uint8)
        cv2.circle(circular_roi, (cX, cY), radius, 255, 10)
        circular_roi = cv2.bitwise_and(thresh_frame, thresh_frame, mask=circular_roi)

        cnts = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnts = imutils.grab_contours(cnts)
        count = 0
        for cnt in cnts:
            (x,y,w,h) = cv2.boundingRect(cnt)
            out_of_wrist = (cY + (cY * 0.25)) > (y+h)
            # Additional check so that points are not counted outside the hand
            limit_points = ((circumference * 0.25) > cnt.shape[0])
            if out_of_wrist and limit_points:
                count += 1
        return count
		

wait_for_camera_warmup = 3
accumulated_weight = 0.5
background = None
roi_top = 50
roi_bottom = 300
roi_left = 350
roi_right = 100
no_frames_for_running_background_avg = 60
cap = cv2.VideoCapture(0)
time.sleep(wait_for_camera_warmup)
num_frames = 0

while True:
    _, frame = cap.read()
    frame_copy = frame.copy()
    roi = frame[roi_top: roi_bottom, roi_right: roi_left]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_roi = cv2.GaussianBlur(gray_roi, (7,7), 0)
    # 
    if num_frames < no_frames_for_running_background_avg:
        cal_running_avg(gray_roi, accumulated_weight)
        if num_frames <= no_frames_for_running_background_avg - 1:
            cv2.putText(frame_copy, 'Please wait, getting background', (200,200), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 255, 2)
            cv2.imshow('Finger Count', frame_copy)
    else:
        hand = segment(gray_roi, 30)
        if hand is not None:
            thresh, hand_segment = hand
            cv2.drawContours(frame_copy, [hand_segment + (roi_right, roi_top)], -1, (255,0,0), 2)
            fingers = get_fingers_count(thresh, hand_segment)
            cv2.putText(frame_copy, str(fingers), (70,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
            cv2.imshow('thresh', thresh)
    cv2.rectangle(frame_copy, (roi_left, roi_top), (roi_right, roi_bottom), (0,0,255), 5)
    num_frames += 1
    cv2.imshow('Finger Count', frame_copy)
    if cv2.waitKey(1) & 0xff == 27:
        break
cap.release()
cv2.destroyAllWindows()