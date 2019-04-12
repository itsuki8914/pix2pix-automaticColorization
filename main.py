import os,sys,shutil
import tensorflow as tf
import numpy as np
import argparse
import cv2,math,glob,random,time
import time
import matplotlib.pyplot as plt
from model import *
from btgen import BatchGenerator

DATASET_DIR = "data"
VAL_DIR ="val"
SAVE_DIR = "model"
SVIM_DIR = "sample"

def training(loss):
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01,
                                       beta1=0.9,
                                       beta2=0.999)
    train_step = optimizer.minimize(loss)
    return train_step

def tileImage(imgs):
    d = int(math.sqrt(imgs.shape[0]-1))+1
    h = imgs[0].shape[0]
    w = imgs[0].shape[1]
    r = np.zeros((h*d,w*d,3),dtype=np.float32)
    for idx,img in enumerate(imgs):
        idx_y = int(idx/d)
        idx_x = idx-idx_y*d
        r[idx_y*h:(idx_y+1)*h,idx_x*w:(idx_x+1)*w,:] = img
    return r

def foloderLength(folder):
    dir = folder
    paths = os.listdir(dir)
    return len(paths)

def printParam(scope):
    total_parameters = 0
    for variable in tf.trainable_variables(scope=scope):
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print("{} has {} parameters".format(scope, total_parameters))

def main():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    if not os.path.exists(SVIM_DIR):
        os.makedirs(SVIM_DIR)
    img_size =256
    bs = 4
    lr = tf.placeholder(tf.float32)
    lmd = tf.placeholder(tf.float32)

    trans_lr = 2e-4
    trans_lmd = 10
    max_step = 100000

    datalen = foloderLength(DATASET_DIR)
    vallen = foloderLength(VAL_DIR)

    # loading images on training
    batch = BatchGenerator(img_size=img_size,datadir=DATASET_DIR)
    val = BatchGenerator(img_size=img_size, datadir=VAL_DIR)
    id = np.random.choice(range(datalen),bs)

    IN_, OUT_ = batch.getBatch(bs,id)[:4]
    IN_ = (IN_ + 1)*127.5
    IN_ =tileImage(IN_)
    OUT_ = (OUT_ + 1)*127.5
    OUT_ = tileImage(OUT_)
    Z_  = np.concatenate([IN_, OUT_],axis=1)
    cv2.imwrite("input.png",Z_)

    x = tf.placeholder(tf.float32, [bs, img_size, img_size, 3])
    t = tf.placeholder(tf.float32, [bs, img_size, img_size, 3])

    y = buildGenerator(x)
    fake_y = buildDiscriminator(x,y,isTraining=True,nBatch=bs)
    real_y = buildDiscriminator(x,t,reuse=True,isTraining=True,nBatch=bs)

    # sce gan
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_y,labels=tf.ones_like (real_y)))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_y,labels=tf.zeros_like (fake_y)))
    g_loss      = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y,labels=tf.ones_like (y)))

    # ls gan
    #d_loss_real = tf.reduce_mean((real_y-tf.ones_like (real_y))**2)
    #d_loss_fake = tf.reduce_mean((fake_y-tf.zeros_like (fake_y))**2)
    #g_loss  = tf.reduce_mean((fake_y-tf.ones_like (fake_y))**2)

    #variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='g')
    wd_g = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES,scope="Generator")
    wd_d = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES,scope="Discriminator")

    wd_g = tf.reduce_sum(wd_g)
    wd_d = tf.reduce_sum(wd_d)

    L1_loss = tf.reduce_mean(tf.abs(y-t))

    d_loss = d_loss_real + d_loss_fake + wd_d
    g_loss = g_loss + lmd * L1_loss + wd_g

    #L2_loss = tf.nn.l2_loss(y-t)
    pre_loss = lmd * L1_loss + wd_g
    #g_pre = tf.train.AdamOptimizer(1e-3,beta1=0.5).minimize(pre_loss, var_list=[x for x in tf.trainable_variables() if "generator"     in x.name])
    g_opt = tf.train.AdamOptimizer(lr,beta1=0.5).minimize(g_loss, var_list=[x for x in tf.trainable_variables() if "Generator"     in x.name])
    d_opt = tf.train.AdamOptimizer(lr/5,beta1=0.5).minimize(d_loss, var_list=[x for x in tf.trainable_variables() if "Discriminator" in x.name])

    total_parameters = 0
    printParam(scope="Generator")
    printParam(scope="Discriminator")

    start = time.time()

    config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.66))

    sess =tf.Session()
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    summary = tf.summary.merge_all()

    ckpt = tf.train.get_checkpoint_state('model')

    if ckpt: # checkpointがある場合
        last_model = ckpt.model_checkpoint_path # 最後に保存したmodelへのパス
        print ("load " + last_model)
        saver.restore(sess, last_model) # 変数データの読み込み
        print("succeed restore model")
    else:
        print("models were not found")
        init = tf.global_variables_initializer()
        sess.run(init)

    print("%.4e sec took initializing"%(time.time()-start))

    hist = []
    g_hist = []
    d_hist = []

    start = time.time()
    """
    for p in range(10000):
        id = np.random.choice(range(datalen),bs)
        batch_images_x, batch_images_t  = batch.getBatch(bs,id)
        tmp, gen_loss = sess.run([g_pre,pre_loss], feed_dict={
            x: batch_images_x,
            t: batch_images_t
        })
        hist.append(gen_loss)
        print("in step %s, pre_loss =%.4e" %(p, gen_loss))

        if p % 100 == 0:
            out = sess.run(y,feed_dict={
                x:batch_images_x})
            X_ = tileImage(batch_images_x[:4])
            Y_ = tileImage(out[:4])
            Z_ = tileImage(batch_images_t[:4])

            X_ = (X_ + 1)*127.5
            Y_ = (Y_ + 1)*127.5
            Z_ = (Z_ + 1)*127.5
            Z_ = np.concatenate((X_,Y_,Z_), axis=1)
            #print(np.max(X_))
            cv2.imwrite("pre{}.png".format(p),Z_)

            fig = plt.figure()
            ax = fig.add_subplot(111)
            plt.title("Loss")
            plt.grid(which="both")
            plt.yscale("log")
            ax.plot(hist,label="gen_loss")
            plt.xlabel('x{} step'.format(100), fontsize = 16)
            plt.ylabel('loss', fontsize = 16)
            plt.legend(loc = 'upper right')
            plt.savefig("histL2.png")
            plt.close()

    print("%.4e sec took 1000steps" %(time.time()-start))
    """

    for i in range(100001):
        # loading images on training
        id = np.random.choice(range(datalen),bs)
        batch_images_x, batch_images_t  = batch.getBatch(bs,id)

        tmp, dis_loss =sess.run([d_opt,d_loss,], feed_dict={
            x: batch_images_x,
            t: batch_images_t,
            lr:trans_lr,
            lmd:trans_lmd
        })

        tmp, gen_loss, l1 = sess.run([g_opt,g_loss, L1_loss], feed_dict={
            x: batch_images_x,
            t: batch_images_t,
            lr:trans_lr,
            lmd:trans_lmd
        })

        """
        id = np.random.choice(range(datalen),bs)
        batch_images_x, batch_images_t  = batch.getBatch(bs,id,ocp=0.1)
        tmp, gen_loss, l1 = sess.run([g_opt,g_loss, L1_loss], feed_dict={
            x: batch_images_x,
            t: batch_images_t,
            lr:trans_lr,
            lmd:trans_lmd
        })
        """
        if trans_lr > 5e-5:
            trans_lr = trans_lr * 0.99998
        if trans_lmd > 5:
            trans_lmd = trans_lmd *0.9998

        print("in step %s, dis_loss = %.4e, gen_loss = %.4e, l1_loss= %.4e" %(i,dis_loss, gen_loss, l1*trans_lmd))
        g_hist.append(gen_loss)
        d_hist.append(dis_loss)

        if i %100 ==0:
            id = np.random.choice(range(vallen),bs)
            batch_images_x, batch_images_t = val.getBatch(bs,id)
            out = sess.run(y,feed_dict={
                x:batch_images_x})
            X_ = tileImage(batch_images_x[:4])
            Y_ = tileImage(out[:4])
            Z_ = tileImage(batch_images_t[:4])

            X_ = (X_ + 1)*127.5
            Y_ = (Y_ + 1)*127.5
            Z_ = (Z_ + 1)*127.5
            Z_ = np.concatenate((X_,Y_,Z_), axis=1)
            #print(np.max(X_))
            cv2.imwrite("{}/{}.png".format(SVIM_DIR,i),Z_)

            fig = plt.figure(figsize=(8,6), dpi=128)
            ax = fig.add_subplot(111)
            plt.title("Loss")
            plt.grid(which="both")
            plt.yscale("log")
            ax.plot(g_hist,label="gen_loss", linewidth = 0.25)
            ax.plot(d_hist,label="dis_loss", linewidth = 0.25)
            plt.xlabel('step', fontsize = 16)
            plt.ylabel('loss', fontsize = 16)
            plt.legend(loc = 'upper right')
            plt.savefig("hist.png")
            plt.close()

            print("%.4f sec took per 100steps lmd = %.4e, lr = %.4e" %(time.time()-start,trans_lmd,trans_lr))
            start = time.time()

        if i%5000==0 :
            if i>10000:
                loss_1k_old = np.mean(g_hist[-2000:-1000])
                loss_1k_new = np.mean(g_hist[-1000:])
                print("old loss=%.4e , new loss=%.4e"%(loss_1k_old,loss_1k_new))
                if loss_1k_old*2 < loss_1k_new:
                    break

            saver.save(sess,os.path.join(SAVE_DIR,"model.ckpt"),i)

if __name__ == '__main__':
    main()
