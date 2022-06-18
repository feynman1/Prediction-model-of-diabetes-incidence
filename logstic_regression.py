import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# 手写实现逻辑回归
def sigmoid(x):
    return 1/(1 + np.exp(-x))
    pass

# 需要拟合的函数
def predict(x, w):
    return  sigmoid(x.dot(w))
    pass

# 损失函数
def loss(x, w, y):
    p = predict(x, w)
    result = 0 if p ==0 else np.log(p)
    result2= 0 if (1 - p) == 0 else np.log(1 - p)
    return -y*result - (1 - y)*result2
    pass
# 梯度函数
def gradient(x, w, y):
    return x.T.dot(predict(x, w) - y)
    pass

def train(X, w, Y, alpha=0.1, times=1000, tolent=0.00000001):
    loss1 = 0
    t = 0
    while t<times:
        t +=1
        for x, y in zip(X, Y):
            sLoss = loss(x, w, y)
            g1 = gradient(x, w, y)
            w2 = w - alpha*g1
            w = w2
            loss1 += sLoss
            pass
        loss1 = 0
    return w
    pass

# 导入数据
path = './diabetes.csv'
pima = pd.read_csv(path)

# 将包含缺失值的属性用NAN代替
NaN_col_names = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
pima[NaN_col_names] = pima[NaN_col_names].replace(0,np.NaN)

# 计算平均值的函数
def mean(index):
    temp = pima[pima[index].notnull()]
    temp = temp[[index,'Outcome']].groupby(['Outcome'])[[index]].mean().reset_index()
    return temp
# 用平均值替代属性中为0的项
pima.loc[(pima['Outcome'] == 0) & (pima['Glucose'].isnull()) ,'Glucose'] = mean('Glucose')['Glucose'][0]
pima.loc[(pima['Outcome'] == 1) & (pima['Glucose'].isnull()) ,'Glucose'] = mean('Glucose')['Glucose'][1]

pima.loc[(pima['Outcome'] == 0) & (pima['BloodPressure'].isnull()) ,'BloodPressure'] = mean('BloodPressure')['BloodPressure'][0]
pima.loc[(pima['Outcome'] == 1) & (pima['BloodPressure'].isnull()) ,'BloodPressure'] = mean('BloodPressure')['BloodPressure'][1]

pima.loc[(pima['Outcome'] == 0) & (pima['SkinThickness'].isnull()) ,'SkinThickness'] = mean('SkinThickness')['SkinThickness'][0]
pima.loc[(pima['Outcome'] == 1) & (pima['SkinThickness'].isnull()) ,'SkinThickness'] = mean('SkinThickness')['SkinThickness'][1]

pima.loc[(pima['Outcome'] == 0) & (pima['Insulin'].isnull()) ,'Insulin'] = mean('Insulin')['Insulin'][0]
pima.loc[(pima['Outcome'] == 1) & (pima['Insulin'].isnull()) ,'Insulin'] = mean('Insulin')['Insulin'][1]

pima.loc[(pima['Outcome'] == 0) & (pima['BMI'].isnull()) ,'BMI'] = mean('BMI')['BMI'][0]
pima.loc[(pima['Outcome'] == 1) & (pima['BMI'].isnull()) ,'BMI'] = mean('BMI')['BMI'][1]


# 划分训练集、测试集
x_pima = pima.drop(['Outcome','Pregnancies','DiabetesPedigreeFunction','Age','BMI','BloodPressure'],axis=1)
y_pima = pima['Outcome']
train_X, test_X, train_Y, test_Y = train_test_split(x_pima,y_pima,train_size = 0.8,random_state = None,stratify=y_pima)


# 先归一化
train_X = train_X/100
test_X = test_X/100

# 再添加截距项
train_X = np.hstack((np.ones(shape=(len(train_X), 1)), train_X))
test_X = np.hstack((np.ones(shape=(len(test_X), 1)), test_X))


w = np.array([0.1, 0.1, 0.1, 0.1])

# 训练
w = train(train_X, w, train_Y)
print(w)

#计算准确率
num1 = 0
num2 = 0
probabilities = []

test_Y.index = range(len(test_Y))
for i in range(len(test_X)):
    temp = predict(test_X[i],w)
    probabilities.append((temp,test_Y[i]))
    if((temp>=0.31)&(test_Y[i]==1)):
        num1 = num1+1
    elif((temp<0.31)&(test_Y[i]==0)):
        num2 = num2+1

print("准确率：" + str((num1+num2)/len(test_Y)))
print("患病：" + str(num1))
print("未患病：" + str(num2))









