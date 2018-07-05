import datetime
import logging
import os
import time

import cv2
import numpy as np
import tensorflow as tf

import cnn_lstm_otc_ocr
import utils
import helper
import math
import gen_tfrecord

EPOCHS = 3
FLAGS = utils.FLAGS

logger = logging.getLogger('Traing for OCR using CNN+LSTM+CTC')
logger.setLevel(logging.INFO)


def train(train_dir=None, val_dir=None, mode='train'):
    #load dataset
    tfrecords_filename = '/home/youth/DL/CNN_LSTM_CTC_Tensorflow/tfrecords/train.tfrecords'
    filename_queue = tf.train.string_input_producer([tfrecords_filename], num_epochs = EPOCHS, shuffle = True)
    images, names, labels = gen_tfrecord.read_and_decode(filename_queue)
    print images, names, labels
#    b, h, w, c = tf.shape(images)
    shape = np.shape(images)
#    print shape
    seq_len = np.array([12 for _ in range(shape[0])], dtype = np.int64)
    labels = utils.sparse_tuple_from_label(labels)
    
    model = cnn_lstm_otc_ocr.LSTMOCR(mode)
    model.build_graph(images, labels, seq_len)


    num_train_samples = gen_tfrecord.get_size(tfrecords_filename)  # 100000
    num_batches_per_epoch = int(num_train_samples / FLAGS.batch_size)  # example: 100000/100


    with tf.device('/cpu:0'):
        config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=config) as sess:
            init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
            sess.run(init_op)
            
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
            train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
            if FLAGS.restore:
                ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
                if ckpt:
                    # the global_step will restore sa well
                    saver.restore(sess, ckpt)
                    print('restore from the checkpoint{0}'.format(ckpt))
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord = coord)
            
            print('=============================begin training=============================')
            
            for cur_epoch in range(FLAGS.num_epochs):
#                shuffle_idx = np.random.permutation(num_train_samples)
                train_cost = 0
                start_time = time.time()
                batch_time = time.time()

                # the tracing part
                for cur_batch in range(num_batches_per_epoch):
                    if (cur_batch + 1) % 100 == 0:
                        print('batch', cur_batch, ': time', time.time() - batch_time)
                    batch_time = time.time()
                    summary_str, batch_cost, step, _ = \
                        sess.run([model.merged_summay, model.cost, model.global_step,
                                  model.train_op])
                    # calculate the cost
                    train_cost += batch_cost * FLAGS.batch_size

                    train_writer.add_summary(summary_str, step)

                    # save the checkpoint
                    if step % FLAGS.save_steps == 1:
                        if not os.path.isdir(FLAGS.checkpoint_dir):
                            os.mkdir(FLAGS.checkpoint_dir)
                        logger.info('save the checkpoint of{0}', format(step))
                        saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'ocr-model'),
                                   global_step=step)
#
                    avg_train_cost = train_cost / ((cur_batch + 1) * FLAGS.batch_size)
#
#                        # train_err /= num_train_samples
                    now = datetime.datetime.now()
                    log = "{}/{} {}:{}:{} Epoch {}/{},avg_train_cost = {:.3f}, time = {:.3f}"
                    print(log.format(now.month, now.day, now.hour, now.minute, now.second,
                                     cur_epoch + 1, FLAGS.num_epochs,  avg_train_cost, time.time() - start_time))
            coord.request_stop()
            coord.join(threads)

def infer(root, mode='infer'):
    
    model = cnn_lstm_otc_ocr.LSTMOCR(mode)
    model.build_graph()
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
        ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        if ckpt:
            saver.restore(sess, ckpt)
            print('restore from ckpt{}'.format(ckpt))
        else:
            print('cannot restore')
            
        for img_file in os.listdir(root):
            start_time = time.time()
            img_path = os.path.join(root, img_file)
            print(img_path)
    # imgList = load_img_path('/home/yang/Downloads/FILE/ml/imgs/image_contest_level_1_validate/')
            file_name = img_path.split('/')[-1].split('_')[0]
            imgList = helper.load_img_path(img_path)
            #print(imgList[:5])
        
            total_steps = len(imgList) / FLAGS.batch_size
            sample_num = len(imgList)*3
          
            total_acc = 0
            for curr_step in xrange(total_steps):
                decoded_expression = []
                imgs_input = []
                seq_len_input = []
                imgs_label = []
                for img in imgList[curr_step * FLAGS.batch_size: (curr_step + 1) * FLAGS.batch_size]:
                    
                    label = img.split('_')[-1].split('.')[0]
                    imgs_label.append(label.upper())
                    
                    #print (img)
                    im = cv2.imread(img, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.
                    im = cv2.resize(im, (FLAGS.image_width, FLAGS.image_height))
                    im = np.reshape(im, [FLAGS.image_height, FLAGS.image_width, FLAGS.image_channel])
    
                    def get_input_lens(seqs):
                        length = np.array([FLAGS.max_stepsize for _ in seqs], dtype=np.int64)
    
                        return seqs, length
    
                    inp, seq_len = get_input_lens(np.array([im]))
                    imgs_input.append(im)
                    seq_len_input.append(seq_len)
                
                imgs_input = np.asarray(imgs_input)
                seq_len_input = np.asarray(seq_len_input)
                seq_len_input = np.reshape(seq_len_input, [-1])
    
                feed = {model.inputs: imgs_input,
                        model.seq_len: seq_len_input}
                dense_decoded_code = sess.run(model.dense_decoded, feed)
        
                for item in dense_decoded_code:
                    expression = ''
    
                    for i in item:
                        if i == -1:
                            expression += ''
                        else:
                            expression += utils.decode_maps[i]
    
                    decoded_expression.append(expression)
    
                acc = utils.test_accuracy_calculation(imgs_label,decoded_expression, True )
                total_acc += acc
            print (total_acc/total_steps)
            print (file_name)
            print (sample_num)
            with open('./result.txt', 'a') as f:
                    f.write(file_name  + ',' + str(round(total_acc/total_steps,2))  + ',' +  str(sample_num) + 
                            ',' + str(round((time.time() -start_time)/sample_num, 2))  + '\n')
#

def main(_):
    if FLAGS.num_gpus == 0:
        dev = '/cpu:0'
    elif FLAGS.num_gpus == 1:
        dev = '/gpu:0'
    else:
        raise ValueError('Only support 0 or 1 gpu.')

    with tf.device(dev):
        if FLAGS.mode == 'train':
            train(FLAGS.train_dir, FLAGS.val_dir, FLAGS.mode)

        elif FLAGS.mode == 'infer':
            infer(FLAGS.infer_dir, FLAGS.mode)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
