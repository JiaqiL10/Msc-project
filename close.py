import cv2


def closeopration(image, size):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    iClose = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    #cv2.imwrite('close.png', iClose)
    return iClose

if __name__ == "__main__":
    name = 'res_m.jpg'
    image = cv2.imread(name)
    closeopration(image, 9)
