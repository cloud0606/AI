import tensorflow as tf
import pandas as pd
from  tensorflow.examples.tutorials.mnist  import  input_data
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

#mnist = input_data.read_data_sets('./data/', one_hot = True)
def numtolist(num):
    '将标签转化为01数组'
    r1 = np.asarray([0]*21)
    r1[20 - num] = 1
    return r1

def numto2D(a):
    '将标签转化为01数组'
    if a >= 18:
        return np.asarray([0,0,1])
    elif a >= 12:
        return np.asarray([0,1,0])
    else:
        return np.asarray([1,0,0])

def train(hidden_units_size,training_iterations):
    num_classes = 3  # 输出大小
    input_size = 32  # 输入大小
    batch_num = 10
    # hidden_units_size = 60  # 隐藏层节点数量
    # training_iterations = 20

    data = pd.read_csv("student-mat.csv")
    # print(data.head())
    # 输入数据

    # 数据类型转换 - 转换为可计算的数据类型int
    for col in data.columns:
        if (not str(data[col].dtype).startswith("int")):
            # print("Coloum Name ", col, " Type ", data[col].dtype)
            # print("Unique values for ", col, data[col].unique(), "\n")
            values = data[col].unique() # 取出该列所以可能取值
            convertor = dict(zip(values, range(len(values)))) # 定义转换函数
            data[col] = [convertor[item] for item in data[col]] # 将定性数据转为定量的数据
            # print("Coloum Name ", col, " Type ", data[col].dtype)
            # print("Unique values for ", col, data[col].unique(), "\n")

    #print(data.describe()) # 查看数据描述信息，均值,max,min
    print('Data Size ', data.shape)

    training_features = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu','Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime',
           'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery',
           'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc',
           'Walc', 'health', 'absences', 'G1', 'G2']
    label_feature = ['G3']
    selected_feature_data = data

    # 区分训练集和测试集
    num_data = data.shape[0]
    num_input = data.shape[1] - 1
    number_training = int(0.7 * num_data) # 训练数据个数
    number_testing = int(num_data - number_training) # 测试数据个数
    print("number of traning :",number_training)
    print("number of testing :",number_testing)

    # -- 训练数据集 --
    training_data_features = np.array(selected_feature_data.head(number_training)[training_features])
    # 转为4D
    training_data_labels = np.zeros([number_training,num_classes], dtype = float, order = 'C')
    temp_label = np.array(selected_feature_data.head(number_training)[label_feature])

    for i in range(number_training):
        label = temp_label[i] # 取出当前标签
        training_data_labels[i] = numto2D(int(label))

    # print('training:')
    # print(training_data_features[0:5])
    # print(training_data_labels[0:5])

    # -- 测试数据集 --house_info.loc[3:6]
    testing_data_features = np.array(selected_feature_data.loc[number_testing:][training_features])
    # 转为4D
    testing_data_labels = np.zeros([number_training,num_classes], dtype = float, order = 'C')
    temp_label = np.array(selected_feature_data.loc[number_testing:][label_feature])

    for i in range(number_testing):
        label = temp_label[i] # 取出当前标签
        testing_data_labels[i] = numto2D(int(label))

    # ---------------  以上，数据集已经整理好 ---------------

    # print('训练数据集特征:',training_data_features.shape)
    # print('训练数据集标签:',training_data_labels.shape)
    # print('测试数据集特征:',testing_data_features.shape)
    # print('测试数据集标签:',testing_data_labels.shape)

    # 300 20 0.05
    X = tf.placeholder(tf.float32, shape = [None, input_size])
    Y = tf.placeholder(tf.float32, shape = [None, num_classes])

    # W1 = tf.Variable(tf.random_normal ([input_size, num_classes], stddev = 0.1))
    # B1 = tf.Variable(tf.constant (0.1), [num_classes])
    #final_opt = tf.matmul(X, W1) + B1  # 输入层到隐藏层正向传播

    # 含一个隐层
    W1 = tf.Variable(tf.random_normal ([input_size, hidden_units_size], stddev = 0.1))
    B1 = tf.Variable(tf.constant (0.1), [hidden_units_size])
    W2 = tf.Variable(tf.random_normal ([hidden_units_size, num_classes], stddev = 0.1))
    B2 = tf.Variable(tf.constant (0.1), [num_classes])

    hidden_opt = tf.matmul(X, W1) + B1  # 输入层到隐藏层正向传播
    hidden_opt = tf.nn.relu(hidden_opt)  # 激活函数，用于计算节点输出值
    final_opt = tf.matmul(hidden_opt, W2) + B2  # 隐藏层到输出层正向传播


    # 对输出层计算交叉熵损失
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=final_opt))
    # 梯度下降算法，这里使用了反向传播算法用于修改权重，减小损失
    opt = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

    # 初始化变量
    init = tf.global_variables_initializer()
    # 计算准确率
    correct_prediction =tf.equal (tf.argmax (Y, 1), tf.argmax(final_opt, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

    sess = tf.Session ()
    sess.run (init)
    train_acc = []
    test_acc = []
    train_loss = []
    for i in range (training_iterations) :
        batch_input = training_data_features[i*batch_num:i*batch_num+batch_num]
        batch_labels = training_data_labels[i*batch_num:i*batch_num+batch_num]
        # 训练
        training_loss = sess.run ([opt, loss], feed_dict = {X: batch_input, Y: batch_labels})
        train_accuracy = accuracy.eval (session = sess, feed_dict = {X: batch_input,Y: batch_labels})
        test_accuracy = accuracy.eval(session=sess, feed_dict={X: testing_data_features, Y: testing_data_labels})
        train_acc.append(train_accuracy)
        test_acc.append(test_accuracy)
        train_loss.append(training_loss[1])
        print ("step : %d, training accuracy = %2f ,training loss %2f,test_accuracy %2f "% (i, train_accuracy,training_loss[1],test_accuracy))

    return  train_acc,test_acc,train_loss
    # print('testing -----')
    # test_accuracy = accuracy.eval(session=sess, feed_dict={X: testing_data_features, Y: testing_data_labels})
    # print("tesing accuracy = %2f " % (test_accuracy))

def averagenum(num):
    print(num)
    nsum = 0
    for i in range(len(num)):
        nsum += num[i]
    return nsum / len(num)

def lineplot(x_data, y_data, x_label="", y_label="", title=""):
    # Create the plot object
    _, ax = plt.subplots()    # Plot the best fit line, set the linewidth (lw), color and
    # transparency (alpha) of the line
    ax.plot(x_data, y_data, lw = 2, color = '#539caf', alpha = 1)    # Label the axes and provide a title
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    return ax

def plot_trendline1(x_num, y, n):
    ax1 = plt.subplot(2, 2, 1)
    plt.sca(ax1) # 选择子图1
    plt.legend()  # 添加这个才能显示图例
    x = np.linspace(1, x_num, x_num)
    plt.xticks(x)
    parameter = np.polyfit(x, y, n) # 计算趋势线
    y2 = [0]*len(y)
    for i in range(len(y)) :
        y2[i] = parameter[0] * i ** 3 + parameter[1] * i ** 2 + parameter[2] * i + 0.6
    plt.xlabel('training step', color='black')
    plt.ylabel('training accuracy', color='black')
    plt.scatter(x, y,label='training accuracy') # 画散点图
    plt.plot(x, y2, color='g',label='trendline')# 画趋势线
    plt.legend() # 添加这个才能显示图例



def plot_trendline2(x_num, y, n):
    ax2 = plt.subplot(2, 2, 2)
    plt.sca(ax2) # 选择子图2
    x = np.linspace(1, x_num, x_num)
    plt.xticks(x)
    parameter = np.polyfit(x, y, n)
    y2 = [0]*len(y)
    print(y2)
    for i in range(len(y)) :
        y2[i] = parameter[0] * i ** 3 + parameter[1] * i ** 2 + parameter[2] * i + 0.6
    avg = averagenum(y)
    # for i in range(len(y2)):
    #     y2[i] = y2[i] + avg
    print('----- training svg',avg)
    plt.xlabel('training step', color='black')
    plt.ylabel('testing accuracy', color='black')
    plt.scatter(x, y,label='testing accuracy')
    plt.plot(x, y2, color='g',label='trendline')
    plt.legend()




def plot_trendline3(x_num, y, n):
    ax3 = plt.subplot(2, 2, 3)
    plt.sca(ax3) # 选择子图3
    x = np.linspace(1, x_num, x_num)
    plt.xticks(x)
    # plt.bar(x, y, width=0.6, tick_label=x, fc='y',color='blue')
    # plt.xlabel('training step', color='black')
    # plt.ylabel('cross entropy', color='black')
    # plt.legend()
    # plt.show()
    parameter = np.polyfit(x, y, n)
    y2 = [0] * len(y)
    print(y2)
    for i in range(len(y)):
        y2[i] = parameter[0] * i ** 3 + parameter[1] * i ** 2 + parameter[2] * i + 1.2
    avg = averagenum(y)
    # for i in range(len(y2)):
    #     y2[i] = y2[i] + avg
    print('----- training svg', avg)
    plt.xlabel('training step', color='black')
    plt.ylabel('training cross entropy', color='black')
    plt.scatter(x, y, label='training cross entropy')
    plt.plot(x, y2, color='r', label='trendline')
    plt.legend()  # 添加这个才能显示图例
    plt.show()
    # plt.barh(range(len(x)), y, tick_label=x)
    # plt.show()
    # parameter = np.polyfit(x, y, n)
    # y2 = [0]*len(y)
    # print(y2)
    # for i in range(len(y)) :
    #     y2[i] = parameter[0] * i ** 3 + parameter[1] * i ** 2 + parameter[2] * i + 1.0
    # avg = averagenum(y2)
    # for i in range(len(y2)):
    #     y2[i] = y2[i] + avg
    # plt.xlabel('testing step', color='black')
    # plt.ylabel('cross entropy', color='black')
    # plt.scatter(x, y)
    # plt.plot(x, y2, color='g')
    # plt.show()

if __name__  == '__main__':
    train_acc, test_acc, train_loss = train(15, 18)
    # print(train_acc)
    # print(test_acc)
    # print(train_loss)
    # lineplot(range(28), test_acc, x_label="step", y_label="test_acc", title="")
    # lineplot(range(28), train_acc, x_label="step", y_label="train_acc", title="")
    # plt.show(lineplot(range(28), train_loss, x_label="step", y_label="train_acc", title=""))
    plot_trendline1(18, test_acc,3)
    plot_trendline2(18, train_acc, 3)
    plot_trendline3(18, train_loss, 3)
    # dotImg(range(21), test_acc)
    # dotImg(range(21), train_acc)
    # dotImg(range(21), train_loss)

   # choose step17