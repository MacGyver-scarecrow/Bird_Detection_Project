import cv2 as cv

import filter
import detector
import image_extractor

def main():
    video = cv.VideoCapture('../video/test.mp4')

    while(1):
        _, frame = video.read()

        frame = filter.resize(frame)

        result_frame = filter.apply_BS_MOG2(frame)

        image_set = image_extractor.run(frame, result_frame)

        detector.run(image_set)

if __name__ == "__main__":
    main()