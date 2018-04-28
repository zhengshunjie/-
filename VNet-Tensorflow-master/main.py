import tensorflow as tf
import VNet as vn
import numpy as np
from scipy.misc import imread,imresize
from os import  walk
from os.path import join

DATA_DIR = 'D:\\zhengshunjie\\workspace\\deeplab\\组织切片2018.3.29\\清洗过 - 副本\\原图像'
LABLE_DIR= 'D:\\zhengshunjie\\workspace\\deeplab\\组织切片2018.3.29\\清洗过 - 副本\\lable'
BATCH_SIZE = 8
NUM_CLASSES = 2
IMG_HEIGHT = 616
IMG_WIDTH = 539

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

images,lables = read_images(DATA_DIR,LABLE_DIR)

tf_input = tf.placeholder(dtype=tf.uint8, shape=(1, IMG_WIDTH, IMG_HEIGHT, 0, 3))
tf_output= tf.placeholder(dtype=tf.uint8, shape=(1, IMG_HEIGHT, IMG_HEIGHT, 0, 1))

logits = vn.v_net(tf_input,1.0,3)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_output,logits=logits))
train_step = tf.train.ProximalGradientDescentOptimizer(0.1).minimize(loss)

init=tf.global_variables_initializer()

def main():
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(500):
            for batch in range(BATCH_SIZE):
                num=np.random(0,700)
                batch_xs=images[num]
                batch_ys=lables[num]
                sess.run(train_step,feed_dict={tf_input:batch_xs,tf_output:batch_ys})

            print("loss:",loss)


if __name__=='__main__':
    main()

# import tensorflow as tf
# import VNet as vn
#
# import numpy as np
# from os.path import join
# import matplotlib.pyplot as plt
# import convert_to_tfrecords
# BATCH_SIZE = 8
#
# TRAIN_FILE = 'train.tfrecords'
# VALIDATION_FILE = 'validation.tfrecords'
#
#
# NUM_CLASSES = 2
# IMG_HEIGHT = convert_to_tfrecords.IMG_HEIGHT
# IMG_WIDTH = convert_to_tfrecords.IMG_WIDTH
# IMG_CHANNELS = convert_to_tfrecords.IMG_CHANNELS
# IMG_PIXELS = IMG_HEIGHT * IMG_WIDTH * IMG_CHANNELS
#
# NUM_TRAIN = convert_to_tfrecords.NUM_TRAIN
# NUM_VALIDARION = convert_to_tfrecords.NUM_VALIDARION
#
# def read_and_decode(filename_queue):
#
#     reader = tf.TFRecordReader()
#
#     _,serialized_example = reader.read(filename_queue)
#
#     features = tf.parse_single_example(serialized_example,features={
#         'label_raw':tf.FixedLenFeature([],tf.string),
#         'image_raw':tf.FixedLenFeature([],tf.string)
#         })
#
#     image = tf.decode_raw(features['image_raw'],tf.uint8)
#     label = tf.decode_raw(features['label_raw'],tf.uint8)
#
# #    image.set_shape([IMG_PIXELS])
#     image = tf.reshape(image,[IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS])
# #    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
#
# #    label.set_shape([IMG_HEIGHT * IMG_WIDTH * 1])
#     label = tf.reshape(label,[IMG_HEIGHT,IMG_WIDTH,1])
# #    label = tf.cast(label,tf.float32) * (1. / 255) - 0.5
#     return image,label
#
#
# def inputs(data_set,batch_size,num_epochs):
#     if not num_epochs:
#         num_epochs = None
#     if data_set == 'train':
#         file = TRAIN_FILE
#     else:
#         file = VALIDATION_FILE
#
#     with tf.name_scope('input') as scope:
#         filename_queue = tf.train.string_input_producer([file])
#     image,label = read_and_decode(filename_queue)
#
#     return image,label
# def loss_funtion(logits_mat,target_mat):
#     print()
#
#
# #loss
#
# def main():
#     with tf.Session() as sess:
#         images, labels = inputs('train', BATCH_SIZE, 1)
#         plt.imshow(images.eval())
#         plt.show()
# if __name__=='__main__':
#     main()