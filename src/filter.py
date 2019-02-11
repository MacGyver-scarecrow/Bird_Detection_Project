import cv2 as cv
import numpy as np
import background_color_mask

BS_MOG2 = cv.createBackgroundSubtractorMOG2(history=60, varThreshold=500, detectShadows=0)

def rgb(frame):
    return cv.cvtColor(frame, cv.COLOR_BGR2RGB)

def color_extraction(frame, bcea):
    tuning_value = 20 #20정도의 오차는 허용하게 함
    lower_boundary_color = np.array([int(bcea[0] - tuning_value), int(bcea[1] - tuning_value), int(bcea[2] - tuning_value)])
    upper_boundary_color = np.array([255, 255, 255])

    mask = cv.inRange(rgb(frame), lower_boundary_color, upper_boundary_color)

    return cv.bitwise_and(frame, frame, mask=mask)

def resize(frame):
    return cv.resize(frame, (640, 480), interpolation=cv.INTER_LINEAR)

def apply_BS_MOG2(frame, bcea):
    return resize(BS_MOG2.apply(color_extraction(frame, bcea)))