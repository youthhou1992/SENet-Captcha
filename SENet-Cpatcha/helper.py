import numpy as np
import os

def load_img_path(images_path):
    tmp = os.listdir(images_path)
    tmp.sort(key=lambda x: int(x.split('.')[0].split('_')[0]))

    file_names = [os.path.join(images_path, s) for s in tmp]
    
    file_names = np.asarray(file_names)
    return file_names

#imgs_path = '/home/youth/DL/CNN_LSTM_CTC_Tensorflow/imgs/infer/samples1_test/'
#x = load_img_path(imgs_path)          
