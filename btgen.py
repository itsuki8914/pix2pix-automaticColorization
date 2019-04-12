import os,sys,shutil
import numpy as np
import cv2,math,glob,random,time
import time
import matplotlib.pyplot as plt

REGULARIZER_COF = 2e-4

class BatchGenerator:
    def __init__(self, img_size, datadir):
        self.folderPath = datadir
        self.imagePath = glob.glob(self.folderPath+"/*.png")
        #self.orgSize = (218,173)
        self.imgSize = (img_size,img_size)
        assert self.imgSize[0]==self.imgSize[1]


    def add_line(self,img,raw):

        mosaic =img
        h, w, _ = img.shape
        img_for_cnt = np.zeros((h, w), np.uint8)

        count = 0
        max_count = 0
        for i in range(8):
            max_count += np.random.randint(0,5)
        while count<max_count:
            x_e = np.random.randint(0,w//8)
            y_e = np.random.randint(0,h//8)
            x_s = random.randint(0, max(0, w - 1 - x_e))
            y_s = random.randint(0, h - 1 - y_e)

            raw_s = np.array(raw[y_s,x_s],dtype="float32")
            raw_e = np.array(raw[y_s+y_e,x_s+x_e],dtype="float32")
            b_s, g_s, r_s = raw_s
            b_e, g_e, r_e = raw_e

            dist_b = np.abs(b_s - b_e)
            dist_g = np.abs(g_s - g_e)
            dist_r = np.abs(r_s - r_e)

            if dist_b < 16 and dist_g < 16 and dist_r < 16:
                bold = np.random.randint(2,6)
                #print(raw_s,b_s,g_s,r_s)
                img = cv2.line(img,(x_s,y_s),(x_s+x_e,y_s+y_e),(int(b_s),int(g_s),int(r_s)),bold)
                count += 1
            else:
                continue
        return mosaic

    def cannyWithLine(self, img, ocp):

        raw = img

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        occupancy = ocp #np.random.uniform(min_occupancy, max_occupancy)

        h, w, _ = img.shape
        img_for_cnt = np.zeros((h, w), np.uint8)

        img = self.add_line(img, raw)

        return img

    def augment(self,x,y):
        if np.random.rand() > 0.5:
            x = cv2.flip(x, 1)
            y = cv2.flip(y, 1)

        if np.random.rand() > 0.5:
            x = cv2.flip(x, 0)
            y = cv2.flip(y, 0)
        return x, y

    def getBatch(self,nBatch,id,ocp=0.05):
        x   = np.zeros( (nBatch,self.imgSize[0],self.imgSize[1],3), dtype=np.float32)
        y   = np.zeros( (nBatch,self.imgSize[0],self.imgSize[1],3), dtype=np.float32)
        for i,j in enumerate(id):

            occupancy = np.random.uniform(0, ocp)

            img = cv2.imread(self.imagePath[j])
            #print(self.imagePath[j])
            raw = img
            img = cv2.resize(img,self.imgSize)
            raw = cv2.resize(raw,self.imgSize)

            img, raw = self.augment(img, raw)

            img = self.cannyWithLine(img, occupancy)

            img, raw = self.augment(img, raw)

            x[i,:,:,:] = (img - 127.5) / 127.5
            y[i,:,:,:] = (raw - 127.5) / 127.5

        return x, y
