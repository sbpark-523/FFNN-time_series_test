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

# Get data
pre._reader(p_path, p_total_list)
pre._reader(e_path, p_total_list)
print(p_total_list.__len__())
pre._reader(np_path, np_total_list)
pre._reader(o_path, np_total_list)
print(np_total_list.__len__())

# resize
p_total_list = pre._resize(p_total_list)
np_total_list = pre._resize(np_total_list)

# divide
X_training, Y_training, X_validation, Y_validation, X_test, Y_test = pre._shuffleNdivide(p_total_list, np_total_list)
print(X_training.shape,", ",X_validation.shape,", ",X_test.shape,", ",Y_training.shape,", ",Y_validation.shape,", ",Y_test.shape)
X_training = X_training.reshape(-1, INPUT_SIZE, 1)
# Y_training = Y_training.reshape(-1, 2, 1)
X_validation = X_validation.reshape(-1, INPUT_SIZE, 1)
# Y_validation = Y_validation.reshape(-1, 2, 1)
X_test = X_test.reshape(-1, INPUT_SIZE, 1)
# Y_test = Y_test.reshape(-1, 2, 1)
print(X_training.shape,", ",X_validation.shape,", ",X_test.shape)

print('Data constructed!')

def _model(X):
    # 256, 1 --> 256, 128
    L1 = tf.layers.conv1d(inputs=X, filters=16, strides=1, kernel_size=[8], padding='SAME', activation=tf.nn.relu)

    # 256, 128 --> 256, 256
    L2 = tf.layers.conv1d(inputs=L1, filters=32, strides=1, kernel_size=[5], padding='SAME', activation=tf.nn.relu)
    L2p = tf.layers.max_pooling1d(inputs=L2, pool_size=2, strides=2, padding='SAME')

    # 256, 256 --> 256, 128
    L3 = tf.layers.conv1d(inputs=L2p, filters=16, strides=1, kernel_size=[3], padding='SAME', activation=tf.nn.relu)

    # reshape: [-1, 피쳐맵수*피쳐맵 크기]
    flat = tf.reshape(L3, [-1, 16*int(INPUT_SIZE/2)])

    dense4 = tf.layers.dense(inputs=flat, units=512, activation=tf.nn.tanh)

    dense5 = tf.layers.dense(inputs=dense4, units=2, activation=tf.nn.tanh)

    return dense5


def _train(final_layer, label):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=final_layer, labels=label))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    return optimizer, cost

def _test(final_layer, label):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=final_layer, labels=label)) # just for view
    p = tf.nn.softmax(final_layer)
    # define validation function
    h_predict = tf.argmax(p, 1)
    correct_y = tf.argmax(label, 1)
    correct_prediction = tf.equal(tf.argmax(p, 1), tf.argmax(label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return cost, accuracy, correct_y, h_predict

def _main():
    X = tf.placeholder(tf.float32, [None, INPUT_SIZE, 1])
    Y = tf.placeholder(tf.float32, [None, 2])

    hypo = _model(X=X)
    optimizer, cost = _train(final_layer=hypo, label=Y)
    cost, accuracy, correct_y, h_predict = _test(final_layer=hypo, label=Y)

    """Initialize"""
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epoch):
        #Batch processing
        avg_loss = 0.0
        acc = 0.0
        total_batch = int(X_training.__len__() / batch_size)
        for i in range(total_batch):
            batch_xs = X_training[i*batch_size:(i+1)*batch_size]
            batch_ys = Y_training[i*batch_size:(i+1)*batch_size]
            """ Training """
            # feed_dict = {X: batch_xs, Y: batch_ys, keep_prob: [0.1, 0.2, 0.3]}
            feed_dict = {X: batch_xs, Y: batch_ys}
            _, c = sess.run([optimizer, cost], feed_dict=feed_dict)
            avg_loss += c / total_batch
            # print('Batch: ', (i+1), ', cost: ', c)
            # print('----------------------------------')
        if epoch % epoch_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_loss))

        """ Validation """
        if (epoch % 10 == 0) :
            loss_val, acc_val, v_Y, v_p = sess.run([cost, accuracy, correct_y, h_predict], feed_dict={X: X_validation, Y: Y_validation})
            # print(v_Y[:30])
            # print(v_p[:30])
            print('-- validation -- Epoch: %d, Loss: %f, Accuracy: %f'%(epoch, loss_val, acc_val))

    print("learning finished!")

    a, t_loss, t_Y, t_trained = sess.run([accuracy, cost, correct_y, h_predict], feed_dict={X: X_test, Y: Y_test})

    print('Accuracy:', a)
    print(t_loss)
    print('Precision: ', precision_score(t_Y, t_trained))
    print('Recall: ', recall_score(t_Y, t_trained))
    print('F1 score: ', f1_score(t_Y, t_trained))

    sess.close()

_main()
# with open('./check/Y_label.txt','w') as Y_write:
#     y_list = [i for i in t_Y]
#     for item in y_list:
#         Y_write.write(str(item)+'\n')
# with open('./P_label.txt','w') as P_write:
#     p_list = [i for i in t_trained]
#     for item in p_list:
#         P_write.write(str(item)+'\n')

