from time import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy
import skimage
import csv
from scipy import signal, fftpack
from scipy.ndimage import fourier_shift
from scipy.signal import correlate
from skimage import data
from skimage.feature import register_translation
from skimage.feature.register_translation import _upsampled_dft
from n import sp_noise
from edge_collection import generater
from sosi_02 import detector
from close import closeopration
from evaluation import evaluate
def canny(img):
    img = cv2.GaussianBlur(img, (3, 3), 0)
    res = cv2.Canny(img, 50, 150)
    return res
def segmentation(i,img,ground_truth,p,s,l=60,th=0.73):
    path = './seg/'
    generater(img, p)
    img_edge = detector('res.avi',l,th)
    seg_1 = closeopration(img_edge, s)
    seg_1 = cv2.cvtColor(seg_1, cv2.COLOR_BGR2GRAY)
    name_s = path+str(i)+'_seg.png'
    cv2.imwrite(name_s,seg_1)
    img_n = sp_noise(img, p)
    img_canny = canny(img_n)
    cv2.imwrite('can.png',img_canny)
    seg_2 = closeopration(img_canny, s)
    name_c = path + str(i) + '_seg_c.png'
    cv2.imwrite(name_c,seg_2)


    # seg_2 = cv2.cvtColor(seg_2, cv2.COLOR_BGR2GRAY)
    score1 = evaluate(seg_1, ground_truth)
    score2 = evaluate(seg_2, ground_truth)
    print(str(score1) + ' : ' + str(score2))
    return score1,score2

file1 = open(r'1.txt', mode='w+', encoding='UTF-8')
file2 = open(r'2.txt', mode='w+', encoding='UTF-8')

for i in range(37):
    name1 = './image/'+str(i)+'.png'
    name2 = './image/' + str(i) + '_gt.png'
    img = cv2.imread(name1)
    ground_truth = cv2.imread(name2, 0)
    a,b=segmentation(i,img, ground_truth, 0.02, 9)
    file1.write(str(a))
    file2.write(str(b))

file1.close()
file2.close()
