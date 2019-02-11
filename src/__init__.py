import cv2 as cv

import filter
import detector
import image_extractor
import time
import tensorflow as tf

def main():
    video = cv.VideoCapture('../video/test.mp4')
    count = 0
    detector.create_graph()

    while(1):
        print(count)
        count = count + 1

        _, frame = video.read()

        frame = filter.resize(frame)

        result_frame = filter.apply_BS_MOG2(frame)

        cv.imshow('frame', frame)
        cv.imshow('result_fram', result_frame)
        image_set = image_extractor.run(frame, result_frame)

        print("걸러진것 개수는 다음과 같다.")
        print(len(image_set))


        start = time.time()
        if(detector.run(image_set)==0):
            print("detect no")
        print(time.time() - start)

        print("-----------------------------")

        k = cv.waitKey(30) & 0xff
        if k == 27:
            break

    frame.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()