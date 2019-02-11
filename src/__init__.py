import cv2 as cv
import filter
import detector
import image_extractor
import time
import tensorflow as tf
import background_color_mask
from matplotlib import pyplot as plt


def main():
    video = cv.VideoCapture('../video/test.mp4')
    count = 0
    detector.create_graph()

    bcea = [0,0,0]
    while(1):
        # print(count)
        count = count + 1
        _, frame = video.read()

        if((count%60)==1):

            start_t = time.time()
            frame = filter.resize(frame)

            #background_color를 뽑지 않는 것이 좋을 수 도 있겠다는 생각이 듬

            # if ((count % 480) == 1):
            #     bcea = background_color_mask.image_color_cluster(frame)  # 프레임에서 가장 많은 부분을 차지하는 배경색 뽑기

            result_frame = filter.apply_BS_MOG2(frame,bcea)

            cv.imshow('frame', frame)
            cv.imshow('result_fram', result_frame)
            image_set = image_extractor.run(frame, result_frame,count)

            print("걸러진 것의 갯수는 다음과 같다.")
            print(len(image_set))

            if(detector.run(image_set)==0):
                print("none detecting")

            print("걸린시간은 다음과 같습니다.")
            print(time.time() - start_t)
            print("-----------------------------")

            k = cv.waitKey(30) & 0xff
            if k == 27:
                break

    frame.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()