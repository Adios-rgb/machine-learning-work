import cv2
import numpy as np
import pyautogui, time, imutils
from win32api import GetSystemMetrics
import pywinauto
from pywinauto import keyboard
import pydirectinput
from pywinauto.application import Application


wait_for_camera = 3
cap = cv2.VideoCapture(0)
time.sleep(wait_for_camera)
frame_width, frame_height = 640, 480
sw, sh = GetSystemMetrics(0), GetSystemMetrics(1)
aspect_w = sw // frame_width
aspect_h = sh // frame_height

try:
    while True:
        try:
            _, frame = cap.read()
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            '''lower and upper bound for whatever color you want to track 
            might change depending on your background or lighting conditions.
            Feel free to experiment with these values'''
            lower_bound_yellow = np.array([0, 215, 160])
            upper_bound = np.array([255, 255, 255]) 
            
            mask1 = cv2.inRange(hsv, lower_bound_yellow, upper_bound)
            kernel = np.ones((3,3), dtype=np.uint8)
            
            # Ímage processing required to clean any noise in the image after masking
            dil1 = cv2.dilate(mask1, kernel, iterations=3)
            opening1 = cv2.morphologyEx(dil1, cv2.MORPH_OPEN, kernel)
            closing1 = cv2.morphologyEx(opening1, cv2.MORPH_CLOSE, kernel)
            
            cnts1 = cv2.findContours(closing1, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cnts1 = imutils.grab_contours(cnts1)
                        
            if len(cnts1) > 0:
                # Using max contour to eliminate the chances of capturing any small noisy contours
                max_area_cnt = max(cnts1, key = cv2.contourArea)
                x, y, w, h = cv2.boundingRect(max_area_cnt)
                x = sw - (x * aspect_w)
                if x < (sw // 2 - 300) and cv2.contourArea(max_area_cnt) > 200:
                    cv2.drawContours(frame, [max_area_cnt], -1, (0,0,255), 2)
                    pydirectinput.keyDown('left')
                elif x > (sw // 2 + 300) and cv2.contourArea(max_area_cnt) > 200:
                    cv2.drawContours(frame, [max_area_cnt], -1, (0,0,255), 2)
                    pydirectinput.keyDown('right')
                else:
                    pydirectinput.keyUp('left')
                    pydirectinput.keyUp('right')
                cv2.line(frame, (sw // (aspect_w * 2) - 150, 0), (sw // (aspect_w * 2) - 150, sh // aspect_h), (0,128,0), 1)
                cv2.line(frame, (sw // (aspect_w * 2) + 150, 0), (sw // (aspect_w * 2) + 150, sh // aspect_h), (0,128,128), 1)

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xff == 27:
                break
        except Exception as e:
            print(str(e))
            continue
except Exception as e:
    print(str(e))

cap.release()
cv2.destroyAllWindows()