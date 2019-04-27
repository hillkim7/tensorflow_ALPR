"""
This program is to recognition of number digits images.
This program has two modes, train and test.
Run "train" mode to make model file and then run test mode.
"""
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import numpy as np
import sys, os
import cv2
import img_utils # image file loader

if len(sys.argv) < 2 or sys.argv[1] not in ('train', 'test'):
  sys.exit('Usage: %s [train|test]' % sys.argv[0])

is_train = (sys.argv[1] == 'train')

model_file_path = './my_models/number_digits'

if is_train and not os.path.isdir('./my_models'):
  os.mkdir('./my_models')

# hyper parameters
learning_rate = 0.001 #0.001
training_epochs = 100 #15
batch_size = 100

#25 35
x_size=25   #실제 이미지 사이즈 25 *35 => 배열 n ,35 ,25
y_size=35

# dropout (keep_prob) rate  0.7~0.5 on training, but should be 1 for testing
keep_prob = tf.placeholder(tf.float32)

# input place holders
X = tf.placeholder(tf.float32, [None, y_size * x_size])
X_img = tf.reshape(X, [-1, y_size , x_size, 1])   # img 28x28x1 (black/white) => 35x25x1 (black/white)
Y = tf.placeholder(tf.float32, [None, 10])

# L1 ImgIn shape=(?, 35, 25, 1)
W1 =  tf.get_variable('w1', shape=[3, 3, 1, 32],initializer=tf.contrib.layers.xavier_initializer(uniform=False))
# W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
#    Conv     -> (?, 35, 25, 32)
#    Pool     -> (?, 18, 13, 32)
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)
'''
Tensor("Conv2D:0", shape=(?, 35, 25, 32), dtype=float32)
Tensor("Relu:0", shape=(?, 35, 25, 32), dtype=float32)
Tensor("MaxPool:0", shape=(?, 18, 13, 32), dtype=float32)
Tensor("dropout/mul:0", shape=(?, 18, 13, 32), dtype=float32)
'''

# L2 ImgIn shape=(?, 18, 13, 32)
W2 =  tf.get_variable('w2', shape=[3, 3, 32, 64], initializer=tf.contrib.layers.xavier_initializer(uniform=False))
# W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
#    Conv      ->(?, 18, 13, 64)
#    Pool      ->(?, 9, 7, 64)
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)
'''
Tensor("Conv2D_1:0", shape=(?, 18, 13, 64), dtype=float32)
Tensor("Relu_1:0", shape=(?, 18, 13, 64), dtype=float32)
Tensor("MaxPool_1:0", shape=(?, 9, 7, 64), dtype=float32)
Tensor("dropout_1/mul:0", shape=(?, 9, 7, 64), dtype=float32)
'''

# L3 ImgIn shape=(?, 9, 7, 64)
W3 =  tf.get_variable('w3', shape=[3, 3, 64, 128],initializer=tf.contrib.layers.xavier_initializer(uniform=False))
# W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
#    Conv      ->(?, 9, 7, 128)
#    Pool      ->(?, 5, 4, 128)
#    Reshape   ->(?, 5 * 4 * 128) # Flatten them for FC
L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[
                    1, 2, 2, 1], padding='SAME')
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)
L3_flat = tf.reshape(L3, [-1, 128 * 5 * 4])
'''
Tensor("Conv2D_2:0", shape=(?, 9, 7, 128), dtype=float32)
Tensor("Relu_2:0", shape=(?, 9, 7, 128), dtype=float32)
Tensor("MaxPool_2:0", shape=(?, 5, 4, 128), dtype=float32)
Tensor("dropout_2/mul:0", shape=(?, 5, 4, 128), dtype=float32)
Tensor("Reshape_1:0", shape=(?, 2560), dtype=float32)
'''

# L4 FC 4x4x128 inputs -> 625 outputs
W4 = tf.get_variable("W4", shape=[128 * 5 * 4, 625],
                     initializer=tf.contrib.layers.xavier_initializer(uniform=False))
# b4 = tf.Variable(tf.random_normal([625]))
b4 = tf.Variable(tf.zeros([625]))

L4 = tf.nn.relu(tf.matmul(L3_flat, W4) + b4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)
'''
Tensor("Relu_3:0", shape=(?, 625), dtype=float32)
Tensor("dropout_3/mul:0", shape=(?, 625), dtype=float32)
'''

# L5 Final FC 625 inputs -> 10 outputs
W5 = tf.get_variable("W5", shape=[625, 10],
                     initializer=tf.contrib.layers.xavier_initializer(uniform=False))
# b5 = tf.Variable(tf.random_normal([10]))
b5 = tf.Variable(tf.zeros([10]))
logits = tf.matmul(L4, W5) + b5
'''
Tensor("add_1:0", shape=(?, 10), dtype=float32)
'''

# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

if is_train:
    # train_data, train_labels = Load_mnist.read_data()
    train_data, train_labels = img_utils.read_data_from_img('my_images/number_data',x_size,y_size)
    print(train_data.shape) #n 35 25
    print(train_labels.shape) # n

    # initialize
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    # train my model
    print('Learning started. It takes sometime.')
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(len(train_data) / batch_size  )

        for i in range(total_batch):
            batch_xs = img_utils.batch1(train_data, i, batch_size,x_size,y_size)
            batch_ys = img_utils.batch2(train_labels,i ,batch_size)
            # print('i:',i,'  total_batch: ',total_batch,'len(train_data): ',len(train_data))
            # print('batch_xs.shape :',batch_xs.shape)  #100, 784
            # print('type(batch_xs) :', type(batch_xs))  #
            # print('batch_ys.shape :',batch_ys.shape)  # 100,10
            # print('type(batch_ys) :', type(batch_ys))  #
            # sys.exit()
            feed_dict = {X: batch_xs, Y: batch_ys, keep_prob: 0.7}
            c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
            avg_cost += c / total_batch

        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
        t = np.int32(np.random.rand(1000))
        np.random.seed(t)
        np.random.shuffle(train_data)
        np.random.seed(t)
        np.random.shuffle(train_labels)

    save_path = saver.save(sess, model_file_path)
    print('Training finished: "%s" file generated' % model_file_path)
    sess.close()

if not is_train:
    # test_data, test_labels = Load_mnist.read_test()  # 저장된 gz에서 읽기
    # test_data, test_labels = Load_mnist.read_test_from_img('./images/test')
    test_data, test_labels = img_utils.read_data_from_img('my_images/number_test',x_size,y_size)
    print(test_data.shape)  # n ,35,25
    print(test_labels.shape)  # n
    sess = tf.Session()
    # sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    saver.restore(sess, model_file_path)

    test_data = test_data.reshape(-1, y_size * x_size)
    print(test_data.shape)
    test_labels = img_utils.convertToOneHot(test_labels, num_classes=10)
    print(test_labels.shape)

    # Test model and check accuracy
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('Accuracy_test:', sess.run(accuracy, feed_dict={
        X: test_data, Y: test_labels, keep_prob: 1}))

    train_data, train_labels = img_utils.read_data_from_img('my_images/number_data',x_size,y_size)
    train_data = train_data.reshape(-1,y_size*x_size)
    train_labels = img_utils.convertToOneHot(train_labels, num_classes=10)
    print('Accuracy_train:', sess.run(accuracy, feed_dict={
        X: train_data, Y: train_labels, keep_prob: 1}))

    # Get one and predict
    r = random.randint(0, len(test_data) - 1)
    print("Label: ", sess.run(tf.argmax(test_labels[r:r + 1], 1)))
    print("Prediction: ", sess.run(
        tf.argmax(logits, 1), feed_dict={X: test_data[r:r + 1], keep_prob: 1}))

    plt.imshow(test_data[r:r + 1].
              reshape(y_size, x_size), cmap='Greys', interpolation='nearest')
    plt.show()
    sess.close()
