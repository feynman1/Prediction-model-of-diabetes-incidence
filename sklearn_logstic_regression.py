import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

# 导入数据
path = './diabetes.csv'
pima = pd.read_csv(path)

# 将包含缺失值的属性用NAN代替
NaN_col_names = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
pima[NaN_col_names] = pima[NaN_col_names].replace(0,np.NaN)
print(pima.isnull().sum())

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

print(pima.isnull().sum())

# 划分训练集、测试集
x_pima = pima.drop(['Outcome','Pregnancies','DiabetesPedigreeFunction','BloodPressure','Age','BMI'],axis=1)
y_pima = pima['Outcome']
train_X, test_X, train_Y, test_Y = train_test_split(x_pima,y_pima,train_size = 0.8,random_state = None,stratify=y_pima)

# 进行训练、预测
lr = LogisticRegression(max_iter=1000)
lr.fit(train_X,train_Y)
predict = lr.predict(test_X)

# 预测准确度
print(accuracy_score(test_Y, predict))

#输出混淆矩阵
print(confusion_matrix(test_Y, predict))

