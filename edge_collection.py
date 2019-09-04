# Generate synthetic video from static image to add (trans-rotation-noise) vibrations
import cv2
import numpy as np
import random
import scipy
from scipy import signal, fftpack
from scipy.signal import correlate
from n import sp_noise
name = '1.png'
def generater(img,p):

    height, width, layers = img.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    # fourcc = cv2.cv2.CV_FOURCC(*'XVID')
    out = cv2.VideoWriter('res.avi', fourcc, 20.0, (width, height))
    delta = 1
    rot = 0.0
    r = 0


    for frame in range(100):
        if (frame % 4 == 0):
            tx = delta
            ty = 0
        else:
            f = frame % 4
            if f == 1:
                tx = 0
                ty = delta
            # r=0
            elif (f == 2):
                tx = -delta
                ty = 0
            # r=rot
            elif (f == 3):
                tx = 0
                ty = -delta
            # r=rot
        # print frame, tx, ty
        M1 = np.float32([[1, 0, tx], [0, 1, ty]])
        M2 = cv2.getRotationMatrix2D((width / 2, height / 2), r, 1)  # rotations only
        dst = cv2.warpAffine(img, M2, (width, height))
        dst = cv2.warpAffine(dst, M1, (width, height))
        dst = sp_noise(dst,p)
        out.write(dst)
        # out.write (np.random.randint(0, 255, (height,width,3)).astype('uint8')) colored noise
        #cv2.imshow('Image', dst)
        #if cv2.waitKey(10) & 0xFF == ord('q'):
            #break
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    name = '1.png'
    img = cv2.imread(name)
    generater(img,0.02)



