import cv2 as cv

import filter
import detector
import image_extractor
import background_color_mask as b

def main():
    video = cv.VideoCapture('../video/test.mp4')

    detector.create_graph()

    count = 0
    while(1):
        print(count)
        _, frame = video.read()

        frame = filter.resize(frame)

        background_median_color = b.image_color_cluster(frame)
        result_frame = filter.apply_BS_MOG2(frame, background_median_color)

        image_set = image_extractor.run(frame, result_frame)

        detector.run(image_set)


if __name__ == "__main__":
    main()