#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 16:20:45 2018

@author: youth
"""

import tensorflow as tf
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

BATCH_SIZE = 4
CAPACITY = 2000
NUM_THREADS = 4
EPOCHS = 3
charset = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
encode_maps = {}
decode_maps = {}
for i, char in enumerate(charset, 1):
    encode_maps[char] = i
    decode_maps[i] = char

SPACE_INDEX = 0
SPACE_TOKEN = ''
encode_maps[SPACE_TOKEN] = SPACE_INDEX
decode_maps[SPACE_INDEX] = SPACE_TOKEN

def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))

def get_label(img_name):
    code =  img_name.split('/')[-1].split('_')[1].split('.')[0] 
    code = ''.join(code.split())
    code = [SPACE_INDEX if code == SPACE_TOKEN else encode_maps[c] for c in list(code.upper())]
    return code 

def get_size(tfrecords_filename):
    accout = 0
    for record in tf.python_io.tf_record_iterator(tfrecords_filename):
        accout += 1
    return accout
def gen_tfrecord(root_path, tfrecords_filename): 
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)
    height = 64
    width = 192
    for img_name in os.listdir(root_path):
        img_path = os.path.join(root_path, img_name)
        print img_path
        img = cv2.imread(img_path).astype(np.float32)/255
        img = cv2.resize(img, (height, width))
        label = get_label(img_name)
        img_raw = img.tostring()
        example = tf.train.Example(features = tf.train.Features(
                feature = {
                'name': _bytes_feature(img_name),
                'image_raw': _bytes_feature(img_raw),
                'label': _bytes_feature(label)}))
        writer.write(example.SerializeToString())
    writer.close()

def read_and_decode(filename_queue, shuffle_batch = True):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
            serialized_example,
            features = {
               'name':tf.FixedLenFeature([], tf.string),
               'image_raw':tf.FixedLenFeature([], tf.string),
               'label': tf.FixedLenFeature([], tf.string)
                    })
    image = tf.decode_raw(features['image_raw'], tf.float32)
    image = tf.reshape(image, [64,192,3])
    
    name = features['name']
    
    label = features['label']
    if shuffle_batch:
        images, names, labels= tf.train.shuffle_batch([image, name, label],
                                                                        batch_size = BATCH_SIZE,
                                                                        capacity = CAPACITY,
                                                                        num_threads = NUM_THREADS,
                                                                        min_after_dequeue=200)
    #tf.train.shuffle_batch_join
#    if shuffle_batch:
#        images, names, labels, widths, heights = tf.train.shuffle_batch_join([image, name, label, width, height],
#                                                                        batch_size = BATCH_SIZE,
#                                                                        capacity = CAPACITY,
#                                                                        min_after_dequeue=212)
    else:
        images, names, labels= tf.train.batch([image, name, label],
                                                            batch_size = BATCH_SIZE,
                                                            capacity = CAPACITY,
                                                            num_threads = NUM_THREADS)
    return images, names, labels

def main_():
    #root_path = '/home/youth/DL/CNN_LSTM_CTC_Tensorflow/type1_test/'
    tfrecords_filename = '/home/youth/DL/CNN_LSTM_CTC_Tensorflow/tfrecords/train.tfrecords'
    filename_queue = tf.train.string_input_producer([tfrecords_filename], num_epochs = EPOCHS, shuffle = True)
    images, names, labels, widths, heights = read_and_decode(filename_queue)
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord = coord)
        for i in range(1):
            imgs, nms, labs, wids, heis = sess.run([images, names, labels, widths, heights])
            print len(imgs)
            print 'batch' + str(i) + ':'
            for j in range(4):
                print nms[j] + ':' + str(labs[j]) + ' ' + str(wids[j]) + ' ' + str(heis[j])
                img = imgs[j]
                plt.subplot(4,1,j+1)
                plt.imshow(img)
            plt.show()
        coord.request_stop()
        coord.join(threads)
                
        
if __name__ == '__main__':
    main_()