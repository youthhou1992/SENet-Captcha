import os
import numpy as np
import tensorflow as tf
import cv2


# +-* + () + 10 digit + blank + space
#num_classes = 3 + 2 + 10 + 1 + 1
num_classes = 10 + 26 +1 +1

maxPrintLen = 100

restore = False #'whether to restore from the latest checkpoint'
checkpoint_dir = './checkpoint/' # 'the checkpoint dir')
initial_learning_rate =  1e-3 #'inital lr'
#
image_height = 64 #image height')
image_width = 192 # 'image width')
image_channel = 3 # 'image channels as input')
#
max_stepsize = 12 # 'max stepsize in lstm, as well as '
##                                                'the output channels of last layer in CNN')
num_hidden = 256 # 'number of hidden units in lstm')
num_epochs = 500 # 'maximum epochs')
batch_size = 1 # 'the batch_size')
save_steps = 100 # 'the step to save checkpoint')
validation_steps = 500 # 'the step to validation')
#
decay_rate = 0.98 # 'the lr decay rate')
beta1 = 0.9 # 'parameter of adam optimizer beta1')
beta2 = 0.999 # 'adam parameter beta2')
#
decay_steps = 10000 # 'the lr decay_step for optimizer')
momentum =  0.9 # 'the momentum')
#
train_dir = './imgs/train/' # 'the train data dir')
val_dir = './imgs/val/' # 'the val data dir')
infer_dir = './imgs/infer/' # 'the infer data dir')
log_dir = './log' # 'the logging dir')
mode = 'train' # 'train, val or infer')
num_gpus = 0 # 'num of gpus')
#

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

#WIDTH = 192
#HEIGHT = 64
#CHANNEL = 3

class DataIterator:
    def __init__(self, data_dir):
        self.image = []
        self.labels = []
        for root, sub_folder, file_list in os.walk(data_dir):
            for file_path in file_list:
                image_name = os.path.join(root, file_path)
                #print ('hello, world')
                print (image_name)
                img1 = cv2.imdecode(np.fromfile(image_name, dtype=np.uint8), -1)
                img1 = cv2.cvtColor(img1, cv2.COLOR_BGRA2BGR)
                img = np.array(img1, 'f') / 255.0 - 0.5
#                im = cv2.imread(image_name).astype(np.float32)/255.
                # resize to same height, different width will consume time on padding
                im = cv2.resize(img, (image_width, image_height))
                im = np.reshape(im, [image_height, image_width, image_channel])
                #if(len(match) == 4):
#                print(len(match))
                code = image_name.split('/')[-1].split('_')[1].split('.')[0]#.upper()
                self.image.append(im)
                #print (code)
                code = [SPACE_INDEX if code == SPACE_TOKEN else encode_maps[c] for c in list(code.upper())]
                self.labels.append(code)

    @property
    def size(self):
        return len(self.labels)

    def the_label(self, indexs):
        labels = []
        for i in indexs:
            labels.append(self.labels[i])

        return labels

    def input_index_generate_batch(self, index=None):
        if index:
            image_batch = [self.image[i] for i in index]
            label_batch = [self.labels[i] for i in index]
        else:
            image_batch = self.image
            label_batch = self.labels

        def get_input_lens(sequences):
            # 64 is the output channels of the last layer of CNN
            lengths = np.asarray([12 for _ in sequences], dtype=np.int64)

            return sequences, lengths

        batch_inputs, batch_seq_len = get_input_lens(np.array(image_batch))
        batch_labels = sparse_tuple_from_label(label_batch)

        return batch_inputs, batch_seq_len, batch_labels

#original_seq:
def train_accuracy_calculation(original_seq, decoded_seq, ignore_value=-1, isPrint=False):
    if len(original_seq) != len(decoded_seq):
        print('original lengths is different from the decoded_seq, please check again')
        return 0
    count = 0
    for i, origin_label in enumerate(original_seq):
        decoded_label = [j for j in decoded_seq[i] if j != ignore_value]
        if isPrint and i < maxPrintLen:
            # print('seq{0:4d}: origin: {1} decoded:{2}'.format(i, origin_label, decoded_label))

            with open('./test.csv', 'w') as f:
                f.write(str(origin_label) + '\t' + str(decoded_label))
                f.write('\n')

        if origin_label == decoded_label:
            count += 1

    return count * 1.0 / len(original_seq)

#original_seq:raw labels,like '2343'
def test_accuracy_calculation(original_seq, decoded_seq,isPrint=False):
    if len(original_seq) != len(decoded_seq):
        print('original lengths is different from the decoded_seq, please check again')
        return 0
    count = 0
    for i, origin_label in enumerate(original_seq):
        #decoded_label = [j for j in decoded_seq[i] if j != ignore_value]
        decoded_label = decoded_seq[i]
        if isPrint and i < maxPrintLen:
            # print('seq{0:4d}: origin: {1} decoded:{2}'.format(i, origin_label, decoded_label))

            with open('./test.csv', 'a') as f:               
                f.write(str(origin_label) + '\t' + str(decoded_label))
                f.write('\n')

        if origin_label == decoded_label:
            count += 1

    return count * 1.0 / len(original_seq)


def sparse_tuple_from_label(sequences, dtype=np.int32):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape


def eval_expression(encoded_list):
    """
    :param encoded_list:
    :return:
    """

    eval_rs = []
    for item in encoded_list:
        try:
            rs = str(eval(item))
            eval_rs.append(rs)
        except:
            eval_rs.append(item)
            continue

    with open('./result.txt') as f:
        for ith in xrange(len(encoded_list)):
            f.write(encoded_list[ith] + ' ' + eval_rs[ith] + '\n')

    return eval_rs
