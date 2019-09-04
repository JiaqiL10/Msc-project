import cv2
import numpy as np
import random
import scipy
from scipy import signal, fftpack
from scipy.signal import correlate

sig_n = 60

np.seterr(divide='ignore', invalid='ignore')
cap = cv2.VideoCapture("res.avi")
ret, frame = cap.read()
rows, cols = frame.shape[:2]

gray = np.zeros((rows, cols), np.uint8)
pgray = np.zeros((rows, cols), np.uint8)
grad = np.zeros((rows, cols), np.int8)

acc = np.zeros((rows, cols), np.float64)
accu = np.zeros((rows, cols), np.uint8)
# dst   = np.zeros((rows, cols), np.uint8) # for gray image output
dst = np.zeros((rows, cols, 3), dtype="uint8")  # color edges
sr = np.zeros((sig_n-1, rows, cols))

d = 0
arr = []
sg = [None] * (sig_n-1)

# reference signal of edges

ref = np.array([-1.0, 1.0, -6.0, 6.0, -1.0, 1.0, -6.0, 6.0, -1.0, 1.0, -5.0, 4.0, 0.0,1.0, -6.0, 6.0, -1.0, 1.0, -6.0, 6.0, -1.0, 1.0, -5.0, 4.0, 0.0,1.0, -6.0, 6.0, -1.0, 1.0, -6.0, 6.0, -1.0, 1.0, -5.0, 4.0, 0.0,1.0, -6.0, 6.0,-1.0, 1.0, -5.0, 4.0, 0.0,1.0, -6.0, 6.0, -1.0, 1.0, -6.0, 6.0, -1.0, 1.0, -5.0, 4.0, 0.0,1.0, -6.0, 6.0, -1.0, 1.0, -6.0, 6.0, -1.0, 1.0, -5.0, 4.0, 0.0,1.0, -6.0, 6.0, -1.0, 1.0, -6.0, 6.0, -1.0, 1.0, -5.0, 4.0, 0.0,1.0, -6.0, 6.0, -1.0, 1.0, -6.0, 6.0, -1.0, 1.0, -5.0, 4.0, 0.0])
ref = ref[0:sig_n-1]
#ref = np.array([-1.0, 1.0, -6.0, 6.0, -1.0, 1.0, -6.0, 6.0, -1.0, 1.0, -5.0, 4.0, 0.0,1.0, -6.0, 6.0, -1.0, 1.0, -6.0, 6.0, -1.0, 1.0, -5.0, 4.0, 0.0,1.0, -6.0, 6.0, -1.0, 1.0, -6.0, 6.0, -1.0, 1.0, -5.0, 4.0, 0.0])
#ref = ref * 3
f = np.fft.fft(ref)
fshift = np.fft.fftshift(f)
y1 = (np.abs(fshift))
Ar = -y1.conjugate()
while (d <= sig_n):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grad = np.int8(gray - pgray) # change the intensity type to iny8
    if (d != 0):
        arr.append(grad)

        accn = np.float64(grad)
        cv2.accumulate(accn, acc)
        cv2.convertScaleAbs(acc, accu)
    pgray = gray
    d += 1

    if (d == sig_n):
        for k in range(d - 1):
            sr[k, :, :] = arr[k]
    if (d == sig_n):
        for v in range(rows):
            for u in range(cols):
                if (accu.item(v, u)) != 0:
                    for k in range(d - 1):
                        sg[k] = sr[k][v][u]
                    # if u==347 and v==615:
                    #	  print sg
                    #   	  print "Hello"

                    f1 = np.fft.fft(sg)
                    fshift1 = np.fft.fftshift(f1)
                    y2 = (np.abs(fshift1))
                    cc = np.abs(np.corrcoef(y2, y1))
                    af = scipy.fft(ref)
                    bf = scipy.fft(sg)
                    c = scipy.ifft(af * scipy.conj(bf))
                    ps = np.argmax(abs(c))
                    ps = ps % 4
                    # print ps
                    if cc[1][0] > 0.75:
                        dst.itemset((v, u, 2), 255)
                        dst.itemset((v, u, 1), 255)
                        dst.itemset((v, u, 0), 255)
    if (d == sig_n):
        # cv2.imshow("Edges",dst)
        cv2.imwrite("edge1.jpg", dst)

    if cv2.waitKey(1) & 0xFF == ord('q') or ret == False:
        cap.release()
        cv2.destroyAllWindows()
        break