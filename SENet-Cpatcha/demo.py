# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 11:30:31 2018

@author: Administrator
"""
import tensorflow as tf
import cv2,os
import cnn_lstm_otc_ocr
import utils
import numpy as np
import time
#CHECKPOINT_DIR = './checkpoint'

TIMESTAMP = 12 

def decode_res(dense_decoded_code):
    decoded_expression = []
    for item in dense_decoded_code:
        expression = ''
        for i in item:
            if i == -1:
                expression += ''
            else:
                expression += utils.decode_maps[i]
        decoded_expression.append(expression)
    return decoded_expression

def fit(model, sess, images):
    imgs_input = []
    seq_len_input = []
    for img in images:
        im = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        im = np.array(im, 'f') /255.0 -0.5
        im = cv2.resize(im, (utils.image_width, utils.image_height))
        im = np.reshape(im, [utils.image_height, utils.image_width, utils.image_channel])
        
        def get_input_lens(seqs):
            length = np.array([TIMESTAMP for _ in seqs], dtype = np.int64)
            return seqs, length
        _, seq_len = get_input_lens(np.array([im]))
        imgs_input.append(im)
        seq_len_input.append(seq_len)
        
    imgs_input = np.asarray(imgs_input)
    seq_len_input = np.asarray(seq_len_input)
    seq_len_input = np.reshape(seq_len_input, [-1])
    
    feed = {model.inputs:imgs_input,
            model.seq_len: seq_len_input,
            model.training_flag: False}
    decode_beg = time.time()
    dense_decoded_code = sess.run(model.dense_decoded, feed)
    decode_end = time.time()
    res = decode_res(dense_decoded_code)
    print(decode_end - decode_beg)
    return res

root = '/home/youth/DL/SENet-Cpatcha/imgs/infer/柳州_test'
img_list = []
read_img_begin = time.time()
for image in os.listdir(root):
    image_name = os.path.join(root, image)
    img = cv2.imdecode(np.fromfile(image_name, dtype=np.uint8), -1)
    img_list.append(img)
read_img_end = time.time()
#print len(img_list)
#with tf.device('/gpus:0'):
#build_model_begin = time.time()
#model = cnn_lstm_otc_ocr.LSTMOCR('infer')
#model.build_graph()
#build_model_end = time.time()

with tf.device('/cpu:0'):
    build_model_begin = time.time()
    model = cnn_lstm_otc_ocr.LSTMOCR('infer')
    model.build_graph()
    build_model_end = time.time()
    config = tf.ConfigProto(allow_soft_placement = True)
    with tf.Session(config = config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables(), max_to_keep = 100)
        ckpt = tf.train.latest_checkpoint(utils.checkpoint_dir)
        if ckpt:
            saver.restore(sess,ckpt)
            print ('restore from ckpt{}'.format(ckpt))
        else:
            print ('cannot restore')
        fit_image_begin = time.time()
        result = fit(model, sess, img_list[:1])
        fit_image_end = time.time()
        print (result)
        print('read img time', read_img_end - read_img_begin)
        print('build model time', build_model_end - build_model_begin)
        print('fit image time', fit_image_end - fit_image_begin)
            
#('read img time', 0.010581016540527344)
#('build model time', 4.76935601234436)
#('fit image time', 29.817568063735962)

        