import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score
import DataReconstruct as input
### tanh
periodic_path = 'D:/DKE/data/Periodic/효종/주기성_데이터.csv'
non_periodic_path = 'D:/DKE/data/Periodic/효종/비주기성_데이터.csv'

learning_rate = 0.01

# Get data
p_data = input.reader(periodic_path)
np_data = input.reader(non_periodic_path)

# Reconstruct for model
X_training, Y_training, X_validation, Y_validation, X_test, Y_test = input.setData2(p_data, np_data)
print(X_training.shape,", ",X_validation.shape,", ",X_test.shape,", ",Y_training.shape,", ",Y_validation.shape,", ",Y_test.shape)

print('Data constructed!')

with open('./check/Y_test.txt','w') as P_write:
    p_list = [i for i in Y_test]
    for item in p_list:
        P_write.write(str(item)+'\n')
with open('./check/Y_vali.txt','w') as P_write:
    p_list = [i for i in Y_test]
    for item in p_list:
        P_write.write(str(item)+'\n')
"""I/O 정의 
    X: 입력 256
    Y: 출력(레이블 수) 2
"""
X = tf.placeholder(tf.float32, [None, 256])
Y = tf.placeholder(tf.float32, [None, 2])
keep_prob = tf.placeholder(tf.float32)  #0.1, 0.2, 0.3

# input
W1 = tf.get_variable("weight1", shape=[256, 200], initializer=tf.contrib.layers.xavier_initializer())
# W1 = tf.Variable(tf.random_normal([256, 200]), name="weight1")
b1 = tf.Variable(tf.random_normal([200]))
L1 = tf.matmul(X, W1) + b1
L1 = tf.nn.dropout(L1, keep_prob[0])

"""hidden Layers
    dropout: 
"""
W2 = tf.get_variable("weight2", shape=[200, 200], initializer=tf.contrib.layers.xavier_initializer())
# W2 = tf.Variable(tf.random_normal([200, 200]), name="weight2")
b2 = tf.Variable(tf.random_normal([200]))
L2 = tf.nn.tanh(tf.matmul(L1, W2) + b2)
L2 = tf.nn.dropout(L2, keep_prob[1])

# W3 = tf.Variable(tf.random_normal([300, 300]), name="weight3")
# W3 = tf.Variable(tf.random_normal([200, 2]), name="weight3")
W3 = tf.get_variable("weight3", shape=[200, 2], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([2]))
L3 = tf.nn.tanh(tf.matmul(L2, W3) + b3)
# L3 = tf.nn.dropout(L3, keep_prob[1])
hypothesis = tf.nn.dropout(L3, keep_prob[2])

# W4 = tf.Variable(tf.random_normal([300, 2]), name="weight4")
# b4 = tf.Variable(tf.random_normal([2]))
# L4 = tf.nn.tanh(tf.matmul(L3, W4) + b4)
# hypothesis = tf.nn.dropout(L4, keep_prob[2])

p = tf.nn.softmax(hypothesis)

param_list = [W1, W2, W3, b1, b2, b3]

saver = tf.train.Saver(param_list)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# define validation function
h_predict = tf.argmax(p, 1)
correct_y = tf.argmax(Y, 1)
correct_prediction = tf.equal(tf.argmax(p, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.import_meta_graph("./save/mytest.ckpt.meta")
saver.restore(sess, tf.train.latest_checkpoint("./save"))
# saver.restore(sess, 'mytest.ckpt')

# saver = tf.train.Saver()
# saver.restore(sess, './mytest.ckpt')

print("!!!!!!!!!!!")


a, p, t_loss, t_Y, t_trained =sess.run([accuracy, p, cost, correct_y, h_predict], feed_dict={X: X_test, Y: Y_test, keep_prob: [1.0, 1.0, 1.0]})


print('Accuracy:', a)
print(p)
print(t_loss)
print('Precision: ', precision_score(t_Y, t_trained))
print('Recall: ', recall_score(t_Y, t_trained))
print('F1 score: ', f1_score(t_Y, t_trained))

sess.close()