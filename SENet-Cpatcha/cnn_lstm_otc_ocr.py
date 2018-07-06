import tensorflow as tf
import utils
from tensorflow.python.training import moving_averages


#FLAGS = utils.FLAGS
num_classes = utils.num_classes
import senet

class LSTMOCR(object):
    def __init__(self, mode):
        self.mode = mode
        # image
        self.inputs = tf.placeholder(tf.float32, [None, utils.image_height, utils.image_width, utils.image_channel])
        # SparseTensor required by ctc_loss op
        self.labels = tf.sparse_placeholder(tf.int32)
        # 1d array of size [batch_size]
        self.seq_len = tf.placeholder(tf.int32, [None])
        # l2
        self.training_flag = tf.placeholder(tf.bool)
        self._extra_train_ops = []

    def build_graph(self):
        self._build_model()
        self._build_train_op()

        self.merged_summay = tf.summary.merge_all()

    def _build_model(self):
        with tf.variable_scope('cnn'):
#            training_flag = tf.placeholder(tf.bool)
            model = senet.SE_Inception_resnet_v2(self.training_flag)
            x = model.Build_SEnet(self.inputs)
#            print 'after cnn', x
        with tf.variable_scope('blstm'):

            x = tf.transpose(x,[0, 2, 1, 3])
            x = tf.reshape(x, [utils.batch_size, 12, -1])

            x.set_shape([utils.batch_size, 12, 3216])
            x = self._dense_blstm(x)
#            print 'after dense_blstm', x
            x = tf.transpose(x, (1,0,2))
            outputs = tf.reshape(x, [-1, utils.num_hidden*2])
#            print outputs
            W = tf.get_variable(name='W',
                                shape=[utils.num_hidden*2, num_classes],
                                dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable(name='b',
                                shape=[num_classes],
                                dtype=tf.float32,
                                initializer=tf.constant_initializer())

            self.logits = tf.matmul(outputs, W) + b
#            shape = tf.shape(FLAGS.batch_size)
            self.logits = tf.reshape(self.logits, [utils.batch_size, -1, num_classes])
            # Time major
            self.logits = tf.transpose(self.logits, (1, 0, 2))
#            print self.logits

    def _build_train_op(self):
        self.global_step = tf.Variable(0, trainable=False)

        self.loss = tf.nn.ctc_loss(labels=self.labels,
                                   inputs=self.logits,
                                   sequence_length=self.seq_len)
        self.cost = tf.reduce_mean(self.loss)
        tf.summary.scalar('cost', self.cost)

        self.lrn_rate = tf.train.exponential_decay(utils.initial_learning_rate,
                                                   self.global_step,
                                                   utils.decay_steps,
                                                   utils.decay_rate,
                                                   staircase=True)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=utils.initial_learning_rate,
                                                beta1=utils.beta1,
                                                beta2=utils.beta2).minimize(self.loss, global_step=self.global_step)
        train_ops = [self.optimizer] + self._extra_train_ops
        self.train_op = tf.group(*train_ops)

        # Option 2: tf.contrib.ctc.ctc_beam_search_decoder
        # (it's slower but you'll get better results)
        # decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len,merge_repeated=False)
        self.decoded, self.log_prob = tf.nn.ctc_beam_search_decoder(self.logits,
                                                                    self.seq_len,
                                                                    merge_repeated=False)
        self.dense_decoded = tf.sparse_tensor_to_dense(self.decoded[0], default_value=-1)

    def _dense_blstm(self, input):
#        print 'after after cnn', input 
        x = tf.transpose(input, [1,0,2])
        x = tf.reshape(x, [-1, 3216])
        x = tf.split(x, 12)
#        print 'x', x
        out = self._blstm(x, 'blstm1')
#        print out
#        x = senet.Fully_connected(out, 256)
#        print x
#        x = tf.split(x, 12)
        x = self._blstm(x, 'blstm2')
        return x
        
    def _blstm(self, input, name):
        with tf.variable_scope(name):
            lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(utils.num_hidden, forget_bias = 1.0)
            lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(utils.num_hidden, forget_bias = 1.0)
            outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell,
                                                                    lstm_bw_cell, input,
                                                                    dtype = tf.float32)
            return outputs
