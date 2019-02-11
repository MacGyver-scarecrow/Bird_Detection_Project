import cv2 as cv
import numpy as np
import background_color_mask

BS_MOG2 = cv.createBackgroundSubtractorMOG2(history=1, varThreshold=500, detectShadows=0)

def hsv(frame):
    return cv.cvtColor(frame, cv.COLOR_BGR2RGB)

def color_extraction(frame):
    frame = cv.resize(frame, (640, 480), interpolation=cv.INTER_LINEAR)
    bcea = background_color_extraction_array = background_color_mask.image_color_cluster(frame) #프레임에서 가장 많은 부분을 차지하는 배경색 뽑기
    tuning_value = 20 #20정도의 오차는 허용하게 함
    lower_boundary_color = np.array([int(bcea[0]-tuning_value),int(bcea[1]-tuning_value),int(bcea[2]-tuning_value)])
    upper_boundary_color = np.array([255, 255, 255])

    mask = cv.inRange(hsv(frame), lower_boundary_color, upper_boundary_color)

    return cv.bitwise_and(frame, frame, mask=mask)

def resize(frame):
    return cv.resize(frame, (640, 480), interpolation=cv.INTER_LINEAR)

def apply_BS_MOG2(frame):
    return resize(BS_MOG2.apply(color_extraction(frame)))