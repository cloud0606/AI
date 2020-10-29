import input_data
import tensorflow as tf
import input_data

# 数据集导入
mnist = input_data.read_data_sets("C:\\USers\\hyq\\PycharmProjects\\DataAnalysis", one_hot=True)
# 实现回归模型
x = tf.placeholder("float", [None, 784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
# 建立模型
y = tf.nn.softmax(tf.matmul(x,W) + b)
# 计算交叉熵
y_ = tf.placeholder("float", [None,10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
# 实现反向传播算法和梯度下降算法
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
# 初始化变量
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# 评估模型
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))