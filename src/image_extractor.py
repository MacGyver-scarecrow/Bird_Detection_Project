import cv2 as cv


def run(frame, result):
    temp = []

    number_of_object, _, stats, _ = cv.connectedComponentsWithStats(result)

    for i in range(1, number_of_object):
        x, y, width, height, area = stats[i]
        if area > 5:
            object_img = frame[y - 20:y + height + 20, x - 20:x + width + 20]
            if object_img.tobytes() is not b'':
                temp.append(cv.imencode('.jpg', object_img)[1].tostring())

    return temp
