__author__ = 'panyx'

import tensorflow as tf
import cv2
import numpy as np
import os
from tensorflow.python.platform import gfile

# ONU_MODEL_DICT = ['HG8321R', 'F603', 'HS8545M', 'F601', 'HG8010_HG8310']
ONU_MODEL_DICT = ['F603', 'HS8545M', 'F601', 'HG266GT', 'HG8321R_HG8310M_HG8010', 'GM219S', 'G-140W-C']
# TYPE OF MODEL eg:HG8321R or HG8310M or general(general is a common model for all types)
MODEL_TYPE = ''
MODEL_SAVE_PATH = './model/pb/'
MODEL_NAME = ''

default_image_size = 299
num_channels = 3
num_classes = 7

#图像输入层对应的名称-224*224-RGB-3通道
IMG_INPUT_TENSOR_NAME = 'ONU/x-input:0'

#预测结构logtis张量所对应的名称
Y_PRE_TENSOR_NAME = 'ONU/classifier_v3/Logits/SpatialSqueeze:0'
#
IS_TRAINING_TENSOR_NAME = 'ONU/is-training:0'


def load_graph(frozen_graph_filename):

    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="ONU",
            op_dict=None,
            producer_op_list=None
        )

        X = graph.get_tensor_by_name(IMG_INPUT_TENSOR_NAME)
        y_pre = graph.get_tensor_by_name(Y_PRE_TENSOR_NAME)
        is_training = graph.get_tensor_by_name(IS_TRAINING_TENSOR_NAME)

    return graph, X, y_pre, is_training

def load_model():

    MODEL_FILE = 'classifier.pb'

    graph, X, y_pre, is_training = load_graph(os.path.join(MODEL_SAVE_PATH, MODEL_FILE))

    sess = tf.Session(graph=graph)

    y_pre_index = tf.argmax(tf.nn.softmax(y_pre) , 1)

    return X, y_pre_index, is_training, sess


if __name__ == '__main__':

    #独立的测试数据
    test_img_dir_path = '../data_0_7/test/'
    onu_dirs = os.listdir(test_img_dir_path)
    # print(onu_dirs)

    #模型评估
    X, y_pre_index, is_training, sess = load_model()

    precision_count = {}
    #初始化统计信息
    for onu_dir in onu_dirs:
        precision_count[onu_dir] = [0, 0, 0, 0]  #样本数,TF，Precision,误分到该类的样本数

    for onu_dir in onu_dirs:
        # if onu_dir != 'HG8321R':
        #     continue
        for img_file in os.listdir(test_img_dir_path + onu_dir):
            precision_count[onu_dir][0] += 1
            #获取图片内容
            img = cv2.imread(test_img_dir_path + onu_dir + '/' + img_file, 1)
            img = cv2.resize(img, (default_image_size, default_image_size), interpolation=cv2.INTER_CUBIC)
            img = np.reshape(img, [-1, default_image_size, default_image_size, num_channels])

            #自定义分类网络预测分类
            pre_y_index_result = sess.run(y_pre_index, feed_dict={X: img, is_training: False})
            # print(pre_y_index_result)
            # [3]
            # 类标取ONU_MODEL_DICT[pre_y_index_result[0]]

            if ONU_MODEL_DICT[pre_y_index_result[0]] == onu_dir:
                precision_count[onu_dir][1] += 1
            else:
                #展示预测错误样本
                print(test_img_dir_path + onu_dir + "/" + img_file)
                precision_count[ONU_MODEL_DICT[pre_y_index_result[0]]][3] += 1
                # img = cv2.imread(test_img_dir_path+onu_dir+"/"+img_file,1)
                # img = cv2.resize(img, (int(1052/2), int(780/2)), interpolation=cv2.INTER_CUBIC)
                # cv2.putText(img, ONU_MODEL_DICT[pre_y_index_result[0]], (30,30), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0,255,0),1)
                # cv2.imshow('1',img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows

        precision_count[onu_dir][2] = precision_count[onu_dir][1] / precision_count[onu_dir][0]

    print(precision_count)