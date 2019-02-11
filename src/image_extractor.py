import cv2 as cv
import os

def run(frame, result, count):
    temp = []

    nlabels, _, stats, _ = cv.connectedComponentsWithStats(result)

    if not (os.path.isdir("../image_extracted/%s/" % count)):  # 해당 프레임에 대한 폴더 생성
        os.makedirs(os.path.join("../image_extracted/%s" % count))

    str = 0
    for i in range(1, nlabels):
        x, y, width, height, area = stats[i]
        if area > 5:
            imgGrop = frame[y - 20:y + height + 20, x - 20:x + width + 20]
            if imgGrop.tobytes() is not b'':
                str = str + 1
                temp.append(cv.imencode('.jpg', imgGrop)[1].tostring())
                cv.imwrite("../image_extracted/%s/%d.jpg" % (count, str), imgGrop)

    return temp
