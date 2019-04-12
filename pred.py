import os,sys,shutil
import tensorflow as tf
import numpy as np
import argparse
import cv2,math,glob,random,time
import time
import matplotlib.pyplot as plt
from main import *
from model import *
DATASET_DIR = "data"
VAL_DIR ="test"
MODEL_DIR = "model"
OUT_DIR = "coloered"

def main(folder="test"):
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    folder=folder
    files = os.listdir(folder)
    input_size = 1024
    bs =1
    val_size =2
    threshold = 0

    start = time.time()

    x = tf.placeholder(tf.float32, [1, input_size, input_size, 3])
    y = buildGenerator(x,reuse=False)
    #fake_y = buildDiscriminator(y,y,isTraining=True,nBatch=bs)

    sess = tf.Session()
    saver = tf.train.Saver()
    #summary = tf.summary.merge_all()

    ckpt = tf.train.get_checkpoint_state(MODEL_DIR)
    if ckpt: # checkpointがある場合
        #last_model = ckpt.all_model_checkpoint_paths[3]
        last_model = ckpt.model_checkpoint_path # 最後に保存したmodelへのパス
        print ("load " + last_model)

        saver.restore(sess, last_model) # 変数データの読み込み
        print("succeed restore model")

    else:
        init = tf.global_variables_initializer()
        sess.run(init)

    print("%.4e sec took initializing"%(time.time()-start))

    start = time.time()
    #

    files = os.listdir(folder)
    for i in range(len(files)):

        #print(files[i])
        img = cv2.imread("{}/{}".format(folder,files[i]),1)
        print(img.shape)
        #img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #img=cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = (img-127.5)/127.5
        h,w = img.shape[:2]

        input = cv2.resize(img,(input_size,input_size))
        input= input.reshape(1,input_size,input_size,3)
        #input = input.reshape(1,h,w,3)

        out = sess.run(y,feed_dict={x:input})
        out = out.reshape(input_size,input_size,3)

        out = cv2.resize(out,(w,h))


        out = cv2.resize(out, (w, h))
        denorm_o = (out + 1) * 127.5
        cv2.imwrite(OUT_DIR+os.sep+'prediction' + files[i] + '.png', denorm_o)


    print("%.4e sec took for predicting" %(time.time()-start))

if __name__ == '__main__':
    folder = "test"
    try:
        folder = sys.argv[1]
    except:
        pass
    main(folder)
