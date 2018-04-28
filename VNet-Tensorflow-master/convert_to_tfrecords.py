from __future__ import absolute_import,division,print_function

import numpy as np
import tensorflow as tf
import time
from scipy.misc import imread,imresize
from os import  walk
from os.path import join


DATA_DIR = 'D:\\zhengshunjie\\workspace\\deeplab\\组织切片2018.3.29\\清洗过 - 副本\\原图像'
LABLE_DIR= 'D:\\zhengshunjie\\workspace\\deeplab\\组织切片2018.3.29\\清洗过 - 副本\\lable'

IMG_HEIGHT = 616
IMG_WIDTH = 539
IMG_CHANNELS = 3
NUM_TRAIN = 700
NUM_VALIDARION = 299


def read_images(data_path,lable_path):
    data_filenames = next(walk(data_path))[2]
    lable_filenames=next(walk(lable_path))[2]
    data_num_files = len(data_filenames)
    lable_num_files=len(lable_filenames)
    data_images = np.zeros((data_num_files,IMG_HEIGHT,IMG_WIDTH,3),dtype=np.uint8)
    lable_images=np.zeros((lable_num_files,IMG_HEIGHT,IMG_WIDTH),dtype=np.uint8)
    for i in range(len(data_filenames)):
        data_img=imread(join(data_path,data_filenames[i]))
        lable_img=imread(join(lable_path,lable_filenames[i]),mode='L')
        data_images[i]=data_img
        lable_images[i]=lable_img

    return data_images,lable_images



def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert(images,labels,name):

    num = images.shape[0]

    filename = name+'.tfrecords'
    print('Writting',filename)

    writer = tf.python_io.TFRecordWriter(filename)
    for i in range(num):

        img_raw = images[i].tostring()
        lab_raw = labels[i].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'label_raw': _bytes_feature(lab_raw),
            'image_raw': _bytes_feature(img_raw)}))

        writer.write(example.SerializeToString())
    writer.close()
    print('Writting End')

def main(argv):
    print('reading images begin')
    start_time = time.time()
    train_images,train_labels = read_images(DATA_DIR,LABLE_DIR)

    duration = time.time() - start_time
    print("reading images end , cost %d sec" %duration)

    #get validation
    validation_images = train_images[:NUM_VALIDARION,:,:,:]
    validation_labels = train_labels[:NUM_VALIDARION]
    train_images = train_images[NUM_VALIDARION:,:,:,:]
    train_labels = train_labels[NUM_VALIDARION:]

    #convert to tfrecords
    print('convert to tfrecords begin')
    start_time = time.time()
    convert(train_images,train_labels,'train')
    convert(validation_images,validation_labels,'validation')
    duration = time.time() - start_time
    print('convert to tfrecords end , cost %d sec' %duration)

if __name__ == '__main__':
    tf.app.run()