import tensorflow as tf
import csv
import numpy as np
import Preprocess as pre
from sklearn.metrics import precision_score, recall_score, f1_score

p_path = "D:/DKE/data/period_classification/주기성_데이터.csv"
np_path = "D:/DKE/data/period_classification/비주기성_데이터.csv"
o_path = "D:/DKE/data/period_classification/outline(non_periodic).csv"
e_path = "D:/DKE/data/period_classification/ECG(periodic).csv"

p_total_list = []
np_total_list = []


INPUT_SIZE = 1024
HIDDEN_SIZE = 500
OUTPUT_SIZE = 2

learning_rate = 0.01
training_epoch = 1000
batch_size = 200
epoch_step = 10


X = tf.placeholder(tf.float32, [None, INPUT_SIZE, 1])
Y = tf.placeholder(tf.float32, [None, 2])

W1 = tf.Variable(tf.random_normal([16, batch_size, 8], stddev=0.01))
print(W1.shape)
C1 = tf.nn.conv1d(value=X, filters=W1, stride=1, padding='SAME', name='conv1')
C1 = tf.nn.relu(C1)

W2 = tf.Variable(tf.random_normal([batch_size, 16, 8]))
C2 = tf.nn.conv1d(value=C1, filters=W2, stride=1, padding='SAME', name='conv2')

print(C1)
print(C2)