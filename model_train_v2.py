__author__ = 'panyx'
import os
import cv2
import numpy as np
import tensorflow as tf
import math
import random
from sklearn.model_selection import train_test_split
from tensorflow.python.framework import graph_util
from skimage import exposure

from net import inception_v4

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "8,9"

slim = tf.contrib.slim

BATCH_SIZE = 32
TEST_BATCH_SIZE = 32

LEARNING_RATE_BASE = 0.0001
LEARNING_RATE_DECAY = 0.85
# REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 500

# GPU_MEMORY_FRACTION = 0.3

TEST_DATA_SAMPLE_SIZE = 500
TRAIN_DATA_SAMPLE_SIZE = 800

default_image_size = inception_v4.inception_v4.default_image_size

NUM_CLASSES = 17

keep_prob = 0.8

# 继续训练=1
RESTORE = 0


def load_augmentation_data(X_train_array):
    X_train = []
    for img_path in X_train_array:
        img = cv2.imread(img_path)
        img = cv2.resize(img, (int(default_image_size * 1.5), int(default_image_size * 1.5)))
        img = random_flip(img)
        img = random_rotation(img)
        img = random_crop(img)
        img = random_exposure(img)
        img = cv2.resize(img, (default_image_size, default_image_size))
        # 归一化
        img = img / 255.0
        X_train.append(img)

    return np.array(X_train)


def train(X_train, y_train, model_save_path, model_name, x_test=None, y_test=None, num_classes=NUM_CLASSES):
    Xd_num = len(X_train)
    image_size_H = default_image_size
    image_size_W = default_image_size
    num_channels = 3

    # x_test = np.reshape(x_test, [-1, image_size_H, image_size_W, num_channels])

    with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
        X = tf.placeholder(tf.float32, [
            None, image_size_H, image_size_W, num_channels], name='x-input')

        y = tf.placeholder(tf.int64, [None], name='y-input')

        is_training = tf.placeholder(tf.bool, name='is-training')

        k_prob = tf.placeholder(tf.float32)

        print('num_classes : ', num_classes)
        logits, _ = inception_v4.inception_v4(X, num_classes=num_classes, is_training=is_training,
                                              dropout_keep_prob=k_prob)
        global_step = tf.Variable(0, trainable=False)

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(y, num_classes), logits=logits))

        learning_rate = tf.train.exponential_decay(
            LEARNING_RATE_BASE,
            global_step,
            Xd_num / BATCH_SIZE,
            LEARNING_RATE_DECAY)

        correct_prediction = tf.equal(y, tf.argmax(tf.nn.softmax(logits), 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

    saver = tf.train.Saver(max_to_keep=1)
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        if RESTORE == 1:
            ckpt = tf.train.get_checkpoint_state(model_save_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        for e in range(TRAINING_STEPS):
            print("=epoch : %d ==========" % e)

            # shuffle indicies
            train_indicies = np.arange(Xd_num)
            print("==shuffle training data==")
            np.random.shuffle(train_indicies)

            # make sure we iterate over the dataset once

            for i in range(int(math.ceil(Xd_num / BATCH_SIZE)) - 1):
                # generate indicies for the batch
                start_idx = (i * BATCH_SIZE) % Xd_num
                idx = train_indicies[start_idx:start_idx + BATCH_SIZE]

                X_train_batch = load_augmentation_data(X_train[idx])

                X_rs = np.reshape(X_train_batch, [BATCH_SIZE, image_size_H, image_size_W, num_channels])
                # create a feed dictionary for this batch
                feed_dict = {X: X_rs,
                             y: y_train[idx],
                             k_prob: keep_prob,
                             is_training: True}

                _, loss_value, step = sess.run([train_step, loss, global_step], feed_dict=feed_dict)
                print("===batch : %d ,training loss : %g ===" % (i, loss_value))

            if e % 10 == 0:
                test_Xd_num = len(x_test)
                test_indicies = np.arange(test_Xd_num)
                count = 0
                total_acc_value = 0
                for i in range(int(math.ceil(test_Xd_num / TEST_BATCH_SIZE)) - 1):
                    count += 1
                    # generate indicies for the batch
                    test_start_idx = (i * TEST_BATCH_SIZE) % test_Xd_num
                    test_idx = test_indicies[test_start_idx:test_start_idx + TEST_BATCH_SIZE]

                    test_X_rs = np.reshape(x_test[test_idx],
                                           [TEST_BATCH_SIZE, image_size_H, image_size_W, num_channels])

                    acc_value = sess.run(accuracy,
                                         feed_dict={X: test_X_rs, y: y_test[test_idx], k_prob: 1.0, is_training: False})
                    total_acc_value += acc_value

                avg_acc_value = total_acc_value / count

                print("After %d training step(s), test avg_acc_value is %g." % (e, avg_acc_value))

                saver.save(sess, os.path.join(model_save_path, model_name), global_step=e)

                output_graph_def = graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names=[
                    'InceptionV4/Logits/Predictions'])

                with tf.gfile.FastGFile('./model/pb/' + 'classifier.pb', mode='wb') as f:
                    f.write(output_graph_def.SerializeToString())
                # writer=tf.summary.FileWriter("./model/tensorboard/8010",tf.get_default_graph())
                # writer.close()


# def load_pre_img():
#
# def pre_process_img():

def load_img():
    train_path = "/home/kp/project/panyx/classify_data/data_20180420/train_crop/"
    test_path = "/home/kp/project/panyx/classify_data/data_20180420/train_crop/"

    X_train = []
    y_train = []
    X_test = []
    y_test = []
    class_list = os.listdir(train_path)
    for c in range(len(class_list)):
        print(c)
        print(class_list[c])

        # 训练数据
        img_list = os.listdir(train_path + class_list[c])
        # 每一个分类随机获取TEST_DATA_SAMPLE_SIZE张作为测试，避免tensorflow显存OOM，维持数据均衡
        sample_size = len(img_list) if len(img_list) <= TRAIN_DATA_SAMPLE_SIZE else TRAIN_DATA_SAMPLE_SIZE
        train_img_list_random = random.sample(img_list, sample_size)
        for img_name in train_img_list_random:
            # img = cv2.imread(train_path + class_list[c] + '/' + img_name)
            img_path = train_path + class_list[c] + '/' + img_name
            # img = cv2.resize(img, (default_image_size, default_image_size), interpolation=cv2.INTER_CUBIC)
            # 训练数据不直接加载，训练时加载并做数据增强
            X_train.append(img_path)
            y_train.append(c)

        # 测试数据
        img_list = os.listdir(test_path + class_list[c])
        # 每一个分类随机获取TEST_DATA_SAMPLE_SIZE张作为测试，避免tensorflow显存OOM，维持数据均衡
        sample_size = len(img_list) if len(img_list) <= TEST_DATA_SAMPLE_SIZE else TEST_DATA_SAMPLE_SIZE
        test_img_list_random = random.sample(img_list, sample_size)
        for img_name in test_img_list_random:
            # 验证数据直接加载
            img = cv2.imread(test_path + class_list[c] + '/' + img_name)
            img = cv2.resize(img, (default_image_size, default_image_size))
            # 归一化
            img = img / 255.0
            X_test.append(img)
            y_test.append(c)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)

    return X_train, y_train, X_test, y_test


class TrainManager:
    def __init__(self, model_save_path, model_name):
        # self.model_type = model_path
        self.model_save_path = model_save_path
        self.model_name = model_name

    def call_train(self):
        X_train, y_train, X_test, y_test = load_img()
        # X_train,X_test , y_train, y_test = train_test_split(X_train_1, y_train_1, test_size=0.3)
        train(X_train, y_train, self.model_save_path, self.model_name, X_test, y_test)


# 完成图像的左右镜像
def random_flip(image, random_flip=True):
    if random_flip and np.random.choice([True, False]):
        image = np.fliplr(image)  # 左右
    if random_flip and np.random.choice([True, False]):
        image = np.flipud(image)  # 上下
    return image


def random_exposure(image, random_exposure=True):
    if random_exposure and np.random.choice([True, False]):
        e_rate = np.random.uniform(0.5, 1.5)
        image = exposure.adjust_gamma(image, e_rate)
    return image


def random_rotation(image, random_rotation=True):
    if random_rotation and np.random.choice([True, False]):
        w, h = image.shape[1], image.shape[0]
        # 0-180随机产生旋转角度。
        angle = np.random.randint(0, 10)
        RotateMatrix = cv2.getRotationMatrix2D(center=(image.shape[1] / 2, image.shape[0] / 2), angle=angle, scale=0.7)
        # image = cv2.warpAffine(image, RotateMatrix, (w,h), borderValue=(129,137,130))
        image = cv2.warpAffine(image, RotateMatrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return image


def random_crop(image, crop_size=default_image_size, random_crop=True):
    if random_crop and np.random.choice([True, False]):
        if image.shape[1] > crop_size:
            sz1 = image.shape[1] // 2
            sz2 = crop_size // 2
            diff = sz1 - sz2
            (h, v) = (np.random.randint(0, diff + 1), np.random.randint(0, diff + 1))
            image = image[v:(v + crop_size), h:(h + crop_size), :]

    return image


if __name__ == '__main__':
    MODEL_SAVE_PATH = './model/'
    MODEL_NAME = 'classify.ckpt'

    train_manager = TrainManager(MODEL_SAVE_PATH, MODEL_NAME)
    train_manager.call_train()
