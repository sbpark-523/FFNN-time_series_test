import tensorflow as tf
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

learning_rate = 0.0001
training_epoch = 100
batch_size = 100
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

    # width( size of feature map), input channel, output channel
    W1 = tf.Variable(tf.random_normal([8, 1, 3], stddev=0.01))
    C1 = tf.nn.conv1d(value=X, filters=W1, stride=1, padding='SAME', name='conv1')
    C1 = tf.nn.tanh(C1)

    W2 = tf.Variable(tf.random_normal([16, 3, 5]))
    C2 = tf.nn.conv1d(value=C1, filters=W2, stride=1, padding='SAME', name='conv2')
    C2 = tf.nn.tanh(C2)

    W3 = tf.Variable(tf.random_normal([8, 5, 3]))
    C3 = tf.nn.conv1d(value=C2, filters=W3, stride=1, padding='SAME', name='conv3')
    C3 = tf.nn.tanh(C3)

    P1 = tf.layers.max_pooling1d(inputs=C3, pool_size=2, padding='SAME', strides=2)

    flat = tf.reshape(P1, [-1, 512*3])

    W4 = tf.Variable(tf.random_normal([512*3, 200]))
    b4 = tf.Variable(tf.random_normal([200]))
    dense4 = tf.nn.softsign(tf.matmul(flat, W4) + b4)

    W5 = tf.Variable(tf.random_normal([200, 2]))
    b5 = tf.Variable(tf.random_normal([2]))
    dense5 = tf.nn.softsign(tf.matmul(dense4, W5) + b5)

    param_list = [W1, W2, W3, W4, W5, b4, b5]

    saver = tf.train.Saver(param_list)

    return dense5, saver

def _train(hypothesis, label):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=label))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    return optimizer, cost

def _test(hypothesis, label):

    # hypothesis = dense5
    p = tf.nn.softmax(hypothesis)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=label))

    # define validation function
    h_predict = tf.argmax(p, 1)
    correct_y = tf.argmax(label, 1)
    correct_prediction = tf.equal(tf.argmax(p, 1), tf.argmax(label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return cost, accuracy, correct_y, h_predict

def _main():
    X = tf.placeholder(tf.float32, [None, INPUT_SIZE, 1])
    Y = tf.placeholder(tf.float32, [None, OUTPUT_SIZE])

    hypo, model_saver = _model(X=X)
    optimizer, cost = _train(hypothesis=hypo, label=Y)
    cost, accuracy, correct_y, h_predict = _test(hypothesis=hypo, label=Y)

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
            feed_dict = {X: batch_xs, Y: batch_ys}
            _, c = sess.run([optimizer, cost], feed_dict=feed_dict)
            avg_loss += c / total_batch
        if epoch % epoch_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_loss))

        """ Validation """
        if (epoch % 5 == 0) :
            loss_val, acc_val, v_Y, v_p = sess.run([cost, accuracy, correct_y, h_predict], feed_dict={X: X_validation, Y: Y_validation})
            print('-- validation -- Epoch: %d, Loss: %f, Accuracy: %f'%(epoch, loss_val, acc_val))

    save_path = model_saver.save(sess, './saved_model_cnn/cnn_param.ckpt')
    print('save path', save_path)

    sess.close()
    print("learning finished!")

_main()