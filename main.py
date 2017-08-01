import pytesseract
import cv2
import numpy as np
from PIL import Image


def noise_reduction(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.dilate(img, kernel)


def invert(img):
    # convert the image to grayscale and flip the foreground
    # and background to ensure foreground is now "white" and
    # the background is "black"
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(img)

    # threshold the image, setting all foreground pixels to
    # 255 and all background pixels to 0
    #thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    return gray


def blob_detection(img):
    #tmp = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)[1]
    tmp = invert(img)
    params = cv2.SimpleBlobDetector_Params()

    #params.filterByArea = True
    #params.minArea = 10000
    #params.maxArea = 40000

    # Set up the detector with default parameters.
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs.
    keypoints = detector.detect(tmp)
    tmp = invert(tmp)
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(tmp, keypoints, np.array([]), (0, 0, 255),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return im_with_keypoints

def canny_edge(img):
    edges = cv2.Canny(img, 100, 200)

    return edges

def make_blob(img):
    # Median Blur plus thresholding to bleed text. Other possibilities are dilate/erosion
    img = cv2.blur(img, (15, 15))
    img = cv2.threshold(img, 0, 256, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    return img

def remove_skew(img):
    coords = np.column_stack(np.where(img > 0))
    rect = cv2.minAreaRect(coords)
    angle = rect[2]
    print(angle)
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = 90 - angle
    print(angle)
    return(rect[0], rect[1], angle)


def remove_bend(img):
    return cv2.dilate(img, kernel)


def character_recognition(img):
    return pytesseract.image_to_string(img)


def block_detection(img):
    output = cv2.connectedComponents(img, 8, cv2.CV_32S)
    labels = output[0]
    original = output[1].astype(np.uint8)

    tmp = []
    for i in range(1, labels):
        tempo = (original == i)
        coords = np.column_stack(np.where(tempo > 0))
        tmp.append(cv2.minAreaRect(coords))

    out = np.array(tmp)
    return out


if __name__ == "__main__":
    img = cv2.imread('documents/example.png', 0)
    imgCopy = img.copy()
    print('image loaded')

    img = invert(img)
    for i in range(0, 10):
        img = make_blob(img)

    #img = canny_edge(img)

    img = block_detection(img)
    #tmp = cv2.normalize(img[1], None, 0, 255, cv2.NORM_MINMAX)
    #img = noise_reduction(img)


    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    # # Scaling the image to 40%
    # cv2.imwrite('documents/output.png', img)
    print(img[6][0])
    print(img[6][1])
    tmp = cv2.rectangle(None, img[6][0], img[6][1], (255, 0, 0))
    tmp = cv2.resize(tmp, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA)
    cv2.imshow('image', tmp)
    cv2.waitKey(0)




    #cv2.imwrite('documents/invert.png', tmp)

    #img = Image.open('documents/test.png')
    #text = character_recognition(img)
    #print (text)
    #with open("documents/prn10-12_rednoise.txt", "w") as f:
    #    f.write(text)