import numpy as np
import cv2

def evaluate(img1,img2):
    ret, th1 = cv2.threshold(img1, 127, 255, cv2.THRESH_BINARY)
    ret, th2 = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY)
    r1, c1 = th1.shape
    r2, c2 = th2.shape

    for i in range(r2):
        for j in range(c2):
            th2[i][j] = 255 - th2[i][j]

    n = r2 - r1
    add = np.zeros((n, c2))
    th1 = np.r_[th1, add]

    intersection = np.logical_and(th2, th1)
    union = np.logical_or(th2, th1)
    iou_score = np.sum(intersection) / np.sum(union)

    return iou_score

if __name__ == "__main__":
    img1 = cv2.imread('res_canny.jpg', 0)
    img2 = cv2.imread('2.png', 0)
    print('the iou score is '+str(evaluate(img1,img2)))