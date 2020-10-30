from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import datetime
import matplotlib.pyplot as plt 

digits = datasets.load_digits();
# # show data
# plt.gray() 
# for i in range(100):
#     plt.matshow(digits.images[i]) 
#     plt.savefig("data/"+str(i)+".png")
X = digits.data
y = digits.target 
# print(X[0],y[0])

# 分割训练集和测试集
# test_size：样本占比，如果是整数的话就是样本的数量
# random_state：是随机数的种子。
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3., random_state=8) 

estimators = {}
# criterion: gini CART算法中采用的度量标准; entropy 信息增益，是C4.5算法中采用的度量标准
estimators['tree'] = tree.DecisionTreeClassifier(criterion='gini',random_state=8) # 决策树

# n_estimators: 树的数量
# bootstrap: 是否随机有放回
# n_jobs: 可并行运行的数量
estimators['forest'] = RandomForestClassifier(n_estimators=20,criterion='gini',bootstrap=True,n_jobs=2,random_state=8) # 随机森林

# # # 决策树预测
# plt.gray() 
# k = 'tree'
# start_time = datetime.datetime.now()
# estimators[k] = estimators[k].fit(X_train, y_train)
# pred = estimators[k].predict(X_test)
# for i in range(20):
#     plt.matshow(X_test[i].reshape(8,8))
#     plt.savefig('tree/'+str(i)+'.png')
# print(pred[:20])

# # 随机森林预测
# plt.gray() 
# k = 'forest'
# start_time = datetime.datetime.now()
# estimators[k] = estimators[k].fit(X_train, y_train)
# pred = estimators[k].predict(X_test)
# for i in range(20):
#     plt.matshow(X_test[i].reshape(8,8))
#     plt.savefig('forest/'+str(i)+'.png')
# print(pred[:20])

# 决策树与随机森林的比较
for k in estimators.keys():
    start_time = datetime.datetime.now()
    print('----%s----' % k)
    estimators[k] = estimators[k].fit(X_train, y_train)
    pred = estimators[k].predict(X_test)
    print(pred[:10])
    print("%s Score: %0.2f" % (k, estimators[k].score(X_test, y_test)))
    scores = cross_val_score(estimators[k], X_train, y_train,scoring='accuracy' ,cv=10)
    # estimator:估计方法对象(分类器)
    # X：数据特征(Features)
    # y：数据标签(Labels)
    # soring：调用方法(包括accuracy和mean_squared_error等等)
    # cv：几折交叉验证
    print("%s Cross Avg. Score: %0.2f (+/- %0.2f)" % (k, scores.mean(), scores.std() * 2))
    end_time = datetime.datetime.now()
    time_spend = end_time - start_time
    print("%s Time: %0.2f" % (k, time_spend.total_seconds()))