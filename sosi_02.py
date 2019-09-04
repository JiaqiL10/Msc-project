# algorithm to extract edge from synth video using tg after perception of sig var 4 positions
from time import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy
import skimage
from scipy import signal, fftpack
from scipy.ndimage import fourier_shift
from scipy.signal import correlate
from skimage import data
from skimage.feature import register_translation
from skimage.feature.register_translation import _upsampled_dft

np.seterr(divide='ignore', invalid='ignore')
t0 = time()
v_name = 'res.avi'
###########################################################
# prepare and definitions
def detector(v_name,n,th):
    cap = cv2.VideoCapture(v_name)
    ret, frame = cap.read()
    rows, cols = frame.shape[:2]

    # ref_image = frame[:,:,1] # reference is the first frame
    ref_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gray = np.zeros((rows, cols), np.uint8)
    pgray = np.zeros((rows, cols), np.uint8)
    grad = np.zeros((rows, cols), np.int8)
    acc = np.zeros((rows, cols), np.float64)
    accu = np.zeros((rows, cols), np.uint8)
    dmax = n  # number of processed frames
    dst = np.zeros((rows, cols, 3), np.uint8)
    sr = np.zeros((dmax, rows, cols))

    # dst[:] = 255
    d = 0
    arr = []
    sg = [None] * dmax
    # reference signal
    ref = np.array([5.0, 0.0, 0.0, -5.0, 5.0, 0.0, 0.0, -5.0, 5.0, 0.0, 0.0, -5.0, 5.0, 0.0, 0.0, -5.0, 5.0, 0.0, 0.0,
                    -5.0, 5.0, 0.0, 0.0, -5.0, 5.0, 0.0, 0.0, -5.0, 5.0, 0.0, 0.0, -5.0, 5.0, 0.0, 0.0, -5.0, 5.0, 0.0,
                    0.0,
                    -5.0, 5.0, 0.0, 0.0, -5.0, 5.0, 0.0, 0.0, -5.0, 5.0, 0.0, 0.0, -5.0, 5.0, 0.0, 0.0, -5.0, 5.0, 0.0,
                    0.0,
                    -5.0])  # 2 or 3 is mainly good for straight v and h lines and still good for slopes
    ref = ref[0:dmax]
    # ref=np.array([ 0.0, 3.0, 0.0, -3, 0.0, 3.0,0, -3.0,0.0,3.0, 0, -3.0, 0.0, 3.0, 0, -3.0, 0.0, 3.0, 0,-3.0 ])#4 corner model good for corner and sloped lines or curves
    f = np.fft.fft(ref)
    fshift = np.fft.fftshift(f)
    y1 = (np.abs(fshift))
    Ar = -y1.conjugate()
    af = scipy.fft(ref)
    pgray = ref_image
    f = open("signal4p.txt", "w")
    ###############
    # start to process a set of oscillation frames

    while (d <= dmax):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # gray = frame[:,:,1]
        grad = np.int8(gray - pgray)
        # align
        # shift, error, diffphase = register_translation(gray, ref_image, 100) #measure shift
        # if (d!=0) :
        # shift=(-shift[0],-shift[1])
        # offset_image = fourier_shift(np.fft.fft2(grad), shift)
        # grad1 = np.fft.ifft2(offset_image)
        # grad2 = np.real(grad1) #np.abs(grad1)
        arr.append(grad)  # append(np.real(grad1))
        # accn= np.float64(grad2)
        cv2.accumulate(np.uint8(grad), acc)  # (grad2,acc) # accumulated gradients
        cv2.convertScaleAbs(acc, accu)  # get the unsigned acc
        pgray = gray
        '''newimg = cv2.resize(accu, (int(cols / 2), int(rows / 2)))
        cv2.imshow('Image', newimg)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break'''
        d += 1
    ###########
    ### load sr as 3d array
    for k in range(dmax):
        sr[k, :, :] = arr[k]

    for v in range(rows):
        for u in range(cols):
            if (accu.item(v, u)) != 0:  # if accumulated pixel value is non zero
                for k in range(dmax):
                    sg[k] = sr[k][v][u]

                if np.std(sg) > 1.5:
                    f.write('%d %d \n' % (u, v))
                    for l in range(dmax):
                        f.write('%d\n' % sg[l])
                    f1 = np.fft.fft(sg)
                    fshift1 = np.fft.fftshift(f1)
                    y2 = (np.abs(fshift1))
                    cc = np.abs(np.corrcoef(y2, y1))
                    bf = scipy.fft(sg)
                    c = scipy.ifft(af * scipy.conj(bf))
                    ps = np.argmax(abs(c))
                    # get orientation of line and color code it
                    ps = ps % 4  # phase shift over integer no of cycles
                    if cc[1][0] > th:
                        if ps == 0:
                            dst.itemset((v, u, 0), 255)
                            dst.itemset((v, u, 1), 255)
                            dst.itemset((v, u, 2), 255)
                        if ps == 1:
                            dst.itemset((v, u, 0), 255)
                            dst.itemset((v, u, 1), 255)
                            dst.itemset((v, u, 2), 255)
                        if ps == 2:
                            dst.itemset((v, u, 0), 255)
                            dst.itemset((v, u, 1), 255)
                            dst.itemset((v, u, 2), 255)
                        if ps == 3:
                            dst.itemset((v, u, 0), 255)
                            dst.itemset((v, u, 1), 255)
                            dst.itemset((v, u, 2), 255)

    cv2.imwrite("res_m.jpg", dst)
    t2 = time()
    print
    'time is %f' % (t2 - t0)
    cap.release()
    f.close()
    cv2.destroyAllWindows()
    return dst

if __name__ == "__main__":
    detector(v_name,60,0.73)



















