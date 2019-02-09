import cv2 as cv

def run(frame, result):
    temp = []

    nlabels, _, stats, _ = cv.connectedComponentsWithStats(result)

    for i in range(1, nlabels):
        x, y, width, height, area = stats[i]

        if area > 10:
            imgGrop = frame[y - 20:y + height + 20, x - 20:x + width + 20]

            if imgGrop.tobytes() is not b'':
                temp.append(cv.imencode('.jpg', imgGrop)[1].tostring())

    return temp