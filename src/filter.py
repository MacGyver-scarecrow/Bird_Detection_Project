import cv2 as cv
import numpy as np

BS_MOG2 = cv.createBackgroundSubtractorMOG2(history=1, varThreshold=500, detectShadows=0)

def hsv(frame):
    return cv.cvtColor(frame, cv.COLOR_BGR2HSV)

def color_extraction(frame):
    lower_boundary_color = np.array([90, 0, 200])
    upper_boundary_color = np.array([255, 255, 255])

    mask = cv.inRange(hsv(frame), lower_boundary_color, upper_boundary_color)

    return cv.bitwise_and(frame, frame, mask=mask)

def resize(frame):
    return cv.resize(frame, (640, 480), interpolation=cv.INTER_LINEAR)

def apply_BS_MOG2(frame):
    return resize(BS_MOG2.apply(color_extraction(frame)))