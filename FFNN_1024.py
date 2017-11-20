import tensorflow as tf
import csv
import numpy as np
# import DataReconstruct as input
from sklearn.metrics import precision_score, recall_score, f1_score
import Preprocess as pre
import MakePlots as mk

### tanh
p_path = "D:/DKE/data/period_classification/주기성_데이터.csv"
np_path = "D:/DKE/data/period_classification/비주기성_데이터.csv"
o_path = "D:/DKE/data/period_classification/outline(non_periodic).csv"
e_path = "D:/DKE/data/period_classification/ECG(periodic).csv"

p_total_list = []
np_total_list = []


INPUT_SIZE = 1024
HIDDEN_SIZE = 400
OUTPUT_SIZE = 2

learning_rate = 0.0001
training_epoch = 1000
batch_size = 200
epoch_step = 10

loss_value_list_train = []
loss_value_list_valid = []
accuracy_value_list = []

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
    INPUT_SIZE: input neurons
    HIDDEN_SIZE: hidden layer's neurons
    OUTPUT_SIZE: output neurons
"""

def _model(X, keep_prob):
    # input
    W1 = tf.Variable(tf.random_normal([INPUT_SIZE, HIDDEN_SIZE]), name="weight1")
    b1 = tf.Variable(tf.random_normal([HIDDEN_SIZE]))
    L1 = tf.matmul(X, W1) + b1
    L1 = tf.nn.dropout(L1, keep_prob[0])

    """hidden Layers
        dropout: 
    """
    W2 = tf.Variable(tf.random_normal([HIDDEN_SIZE, HIDDEN_SIZE]), name="weight2")
    b2 = tf.Variable(tf.random_normal([HIDDEN_SIZE]))
    L2 = tf.nn.softsign(tf.matmul(L1, W2) + b2)
    L2 = tf.nn.dropout(L2, keep_prob[1])

    # W3 = tf.Variable(tf.random_normal([300, 300]), name="weight3")
    W3 = tf.Variable(tf.random_normal([HIDDEN_SIZE, HIDDEN_SIZE]), name="weight3")
    b3 = tf.Variable(tf.random_normal([HIDDEN_SIZE]))
    L3 = tf.nn.softsign(tf.matmul(L2, W3) + b3)
    L3 = tf.nn.dropout(L3, keep_prob[1])

    W4 = tf.Variable(tf.random_normal([HIDDEN_SIZE, HIDDEN_SIZE]), name="weight4")
    b4 = tf.Variable(tf.random_normal([HIDDEN_SIZE]))
    L4 = tf.nn.softsign(tf.matmul(L3, W4) + b4)
    L4 = tf.nn.dropout(L4, keep_prob[1])

    W5 = tf.Variable(tf.random_normal([HIDDEN_SIZE, HIDDEN_SIZE]), name="weight5")
    b5 = tf.Variable(tf.random_normal([HIDDEN_SIZE]))
    L5 = tf.nn.softsign(tf.matmul(L4, W5) + b5)
    L5 = tf.nn.dropout(L5, keep_prob[1])

    W6 = tf.Variable(tf.random_normal([HIDDEN_SIZE, HIDDEN_SIZE]), name="weight6")
    b6 = tf.Variable(tf.random_normal([HIDDEN_SIZE]))
    L6 = tf.nn.softsign(tf.matmul(L5, W6) + b6)
    L6 = tf.nn.dropout(L3, keep_prob[1])

    W7 = tf.Variable(tf.random_normal([HIDDEN_SIZE, OUTPUT_SIZE]), name="weight7")
    b7 = tf.Variable(tf.random_normal([OUTPUT_SIZE]))
    L7 = tf.nn.softsign(tf.matmul(L6, W7) + b7)
    # L3 = tf.nn.dropout(L3, keep_prob[1])

    hypothesis = tf.nn.dropout(L7, keep_prob[2])

# paramenter
    param_list = [W1, W2, W3, W4, W5, W6, W7, b1, b2, b3, b4, b5, b6, b7]

    saver = tf.train.Saver(param_list)

    return hypothesis, saver


def _train(hypothesis, label):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=label))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    return optimizer, cost


def _test(hypothesis, label):

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=label))
    p = tf.nn.softmax(hypothesis)

    # define validation function
    h_predict = tf.argmax(p, 1)
    correct_y = tf.argmax(label, 1)
    correct_prediction = tf.equal(tf.argmax(p, 1), tf.argmax(label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return cost, accuracy, correct_y, h_predict


def _main():
    # modeling
    X = tf.placeholder(tf.float32, [None, INPUT_SIZE])
    Y = tf.placeholder(tf.float32, [None, OUTPUT_SIZE])
    keep_prob = tf.placeholder(tf.float32)  #0.1, 0.2, 0.3

    hypo, model_saver = _model(X=X, keep_prob=keep_prob)
    optimizer, cost = _train(hypothesis=hypo, label=Y)
    cost, accuracy, correct_y, h_predict = _test(hypothesis=hypo, label=Y)

    """Initialize"""
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    #training

    for epoch in range(training_epoch):
        # Batch processing
        avg_loss = 0.0
        acc = 0.0
        total_batch = int(X_training.__len__() / batch_size)
        for i in range(total_batch):
            batch_xs = X_training[i*batch_size:(i+1)*batch_size]
            batch_ys = Y_training[i*batch_size:(i+1)*batch_size]
            """ Training """
            feed_dict = {X: batch_xs, Y: batch_ys, keep_prob: [1.0, 1.0, 1.0]}
            _, c = sess.run([optimizer, cost], feed_dict=feed_dict)
            avg_loss += c / total_batch
            # print('Batch: ', (i+1), ', cost: ', c)
            # print('----------------------------------')
        if epoch % 1 == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_loss))

        # 매 에폭마다 loss 기록
        loss_value_list_train.append(avg_loss)      # show the result (Training)



        """ Validation """
        if (epoch % 10 == 0) :
            loss_val, acc_val, v_Y, v_p = sess.run([cost, accuracy, correct_y, h_predict], feed_dict={X: X_validation, Y: Y_validation, keep_prob: [1.0, 1.0, 1.0]})
            # print(v_Y[:30])
            # print(v_p[:30])
            print('-- validation -- Epoch: %d, Loss: %f, Accuracy: %f'%(epoch, loss_val, acc_val))
            loss_value_list_valid.append(loss_val)      # show the result (Validation)
            accuracy_value_list.append(acc_val)
            # print('-- validation -- Epoch: %d, Loss: %f, Accuracy: %f'%(epoch, loss_val, acc_val))

    save_path = model_saver.save(sess, "./save_test/mytest.ckpt")
    print('save_path', save_path)

    print("learning finished!")

    a, t_loss, t_Y, t_trained =sess.run([accuracy, cost, correct_y, h_predict], feed_dict={X: X_test, Y: Y_test, keep_prob: [1.0, 1.0, 1.0]})

    print('Accuracy:', a)
    # print(p)
    print(t_loss)
    print('Precision: ', precision_score(t_Y, t_trained))
    print('Recall: ', recall_score(t_Y, t_trained))
    print('F1 score: ', f1_score(t_Y, t_trained))

    sess.close()
_main()


mk._draw_graph(loss_value_list_valid, accuracy_value_list)