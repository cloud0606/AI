import tensorflow as tf
import pandas as pd
import numpy as np
import csv

data = pd.read_csv("student-mat.csv")
# print(data.head())
# 输入数据

def numtolist(num):
    '将标签转化为01数组'
    r1 = np.asarray([0]*21)
    r1[20 - num] = 1
    return r1

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
number_training = int(0.8 * num_data) # 训练数据个数
number_testing = int(num_data - number_training) # 测试数据个数
print("number of traning :",number_training)
print("number of testing :",number_testing)

# -- 训练数据集 --
training_data_features = np.array(selected_feature_data.head(number_training)[training_features])
# 结果数据维度转换为20维
training_data_labels = np.zeros([number_training,21], dtype = float, order = 'C')
temp_label = np.array(selected_feature_data.head(number_training)[label_feature])

for i in range(number_training):
    label = temp_label[i] # 取出当前标签
   # print(label)
  #  print(numtolist(int(label)))
    lst = np.reshape(numtolist(int(label)),(1,21)) # 将一维的0~20取值的分数转为21维的01列表
    training_data_labels[i] = lst

# -- 测试数据集 --
testing_data_features = np.array(selected_feature_data.head(number_testing)[training_features])
# 结果数据维度转换为20维
testing_data_labels = np.zeros([number_testing,21], dtype = float, order = 'C')
temp_label_ = np.array(selected_feature_data.head(number_testing)[label_feature])

for i in range(number_testing):
    label = temp_label[i] # 取出当前标签
  #  print(label)
  #  print(numtolist(int(label)))
    lst = np.reshape(numtolist(int(label)),(1,21)) # 将一维的0~20取值的分数转为21维的01列表
    testing_data_labels[i] = lst

# ---------------  以上，数据集已经整理好 ---------------

print('训练数据集特征:',training_data_features.shape)
print('训练数据集标签:',training_data_labels.shape)
print('测试数据集特征:',testing_data_features.shape)
print('测试数据集标签:',testing_data_labels.shape)

# 构建softmax回归模型
x = tf.placeholder("float", shape=[None, 32]) # 输入变量
y_ = tf.placeholder("float", shape=[None, 21]) # 输出预测值

W = tf.Variable(tf.zeros([32,21]))
b = tf.Variable(tf.zeros([21]))


y = tf.nn.softmax(tf.matmul(x,W) + b) # 回归模型，计算softmax概率值
cross_entropy = -tf.reduce_sum(y_*tf.log(y)) # 损失函数：交叉熵(对数似然函数)
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)# 优化算法：梯度下降法，步长0.01

init = tf.global_variables_initializer() # 根据提示把initialize_all_variables改为global_variables_initializer
sess = tf.Session()
sess.run(init)
# 训练过程
for i in range(31):
#     train_step.run(feed_dict={x: training_data_features, y_: training_data_labels})
    sess.run(train_step,feed_dict={x: training_data_features[i*10:i*10+10], y_: training_data_labels[i*10:i*10+10]})
# print(sess.run(W[0]))
# print(sess.run(b))
# print(training_data_features[:5])
# print(training_data_labels[:5])
# 评估模型
a = tf.argmax(y,1)
aa = tf.argmax(y_,1)
correct_prediction = tf.equal(a, aa)
print(tf.argmax(y,1))
print(tf.argmax(y_,1))
print(tf.equal(tf.argmax(y,1), tf.argmax(y_,1)))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) # 分类准确率
print(sess.run(accuracy,feed_dict={x: testing_data_features, y_: testing_data_labels}))
