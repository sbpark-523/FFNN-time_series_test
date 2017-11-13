import tensorflow as tf
import csv
import numpy as np
import DataReconstruct as input
from sklearn.metrics import precision_score, recall_score, f1_score

### tanh
periodic_path = 'D:/DKE/data/Periodic/효종/주기성_데이터.csv'
non_periodic_path = 'D:/DKE/data/Periodic/효종/비주기성_데이터.csv'

learning_rate = 0.001
training_epoch = 25
batch_size = 200
epoch_step = 1
input_size =256

# Get data
p_data = input.reader(periodic_path)
np_data = input.reader(non_periodic_path)

# Reconstruct for model
X_training, Y_training, X_validation, Y_validation, X_test, Y_test = input.setData2(p_data, np_data)
print(X_training.shape,", ",X_validation.shape,", ",X_test.shape,", ",Y_training.shape,", ",Y_validation.shape,", ",Y_test.shape)
X_training = X_training.reshape(-1, input_size, 1)
# Y_training = Y_training.reshape(-1, 2, 1)
X_validation = X_validation.reshape(-1, input_size, 1)
# Y_validation = Y_validation.reshape(-1, 2, 1)
X_test = X_test.reshape(-1, input_size, 1)
# Y_test = Y_test.reshape(-1, 2, 1)
print(X_training.shape,", ",X_validation.shape,", ",X_test.shape)
print('Data constructed!')


X = tf.placeholder(tf.float32, [None, input_size, 1])
Y = tf.placeholder(tf.float32, [None, 2])

# 256, 1 --> 256, 128
L1 = tf.layers.conv1d(inputs=X, filters=128, strides=1, kernel_size=[8], padding='SAME', activation=tf.nn.relu)

# 256, 128 --> 256, 256
L2 = tf.layers.conv1d(inputs=L1, filters=256, strides=1, kernel_size=[5], padding='SAME', activation=tf.nn.relu)

# 256, 256 --> 256, 128
L3 = tf.layers.conv1d(inputs=L2, filters=128, strides=1, kernel_size=[3], padding='SAME', activation=tf.nn.relu)

# reshape: [-1, 피쳐맵수*피쳐맵 크기]
flat = tf.reshape(L3, [-1, 128*256])

dense4 = tf.layers.dense(inputs=flat, units=512, activation=tf.nn.tanh)

dense5 = tf.layers.dense(inputs=dense4, units=2, activation=tf.nn.tanh)

p = tf.nn.softmax(dense5)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dense5, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# X_in = x_in.reshape(-1, 10, 1)
#
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
#
# _r = sess.run(L1, feed_dict={X: X_in})
#
# print(_r.shape)

# define validation function
h_predict = tf.argmax(p, 1)
correct_y = tf.argmax(Y, 1)
correct_prediction = tf.equal(tf.argmax(p, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

"""Initialize"""
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# batch_xs = X_training[0*batch_size:(0+1)*batch_size]
# batch_ys = Y_training[0*batch_size:(0+1)*batch_size]
# feed_dict = {X: batch_xs, Y: batch_ys}
# x_shape, layer1_shape, layer2_shape, layer3_shape, flat_shape = sess.run([X, L1, L2, L3, flat], feed_dict=feed_dict)
# print(x_shape.shape)
# print(layer1_shape.shape)
# print(layer2_shape.shape)
# print(layer3_shape.shape)
# print(flat_shape.shape)

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
    if (epoch % 5 == 0) :
        loss_val, acc_val, v_Y, v_p = sess.run([cost, accuracy, correct_y, h_predict], feed_dict={X: X_validation, Y: Y_validation})
        # print(v_Y[:30])
        # print(v_p[:30])
        print('-- validation -- Epoch: %d, Loss: %f, Accuracy: %f'%(epoch, loss_val, acc_val))
        # print('-- validation -- Epoch: %d, Loss: %f, Accuracy: %f'%(epoch, loss_val, acc_val))

        # if epoch %10 == 0 : print('Epoch: %d, Loss: %f, Accuracy: %f'%(epoch, avg_loss, acc))

print("learning finished!")

# Test model
correct_prediction = tf.equal(tf.argmax(p, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
a, p, t_loss, t_Y, t_trained = sess.run([accuracy, p, cost, correct_y, h_predict], feed_dict={X: X_test, Y: Y_test})

print('Accuracy:', a)
print(p)
print(t_loss)
print('Precision: ', precision_score(t_Y, t_trained))
print('Recall: ', recall_score(t_Y, t_trained))
print('F1 score: ', f1_score(t_Y, t_trained))


with open('./check/Y_label.txt','w') as Y_write:
    y_list = [i for i in t_Y]
    for item in y_list:
        Y_write.write(str(item)+'\n')
with open('./P_label.txt','w') as P_write:
    p_list = [i for i in t_trained]
    for item in p_list:
        P_write.write(str(item)+'\n')

