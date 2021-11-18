import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection  import train_test_split
import matplotlib.pyplot as plt

def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width','petal length','petal width','label']
    data = np.array(df.iloc[:100, [0,1,-1]])
    for i in range(len(data)):
        if data[i,-1] == 0:
            data[i,-1] = -1

    return data[:,:2], data[:, -1]
class Boost:
    def __init__(self, n_estimators=50, learning_rate = 1.0):
        self.clf_num = n_estimators
        self.learning_rate = learning_rate
    def init_arg(self, datasets, labels):
        self.X = datasets
        self.y = labels
        self.m, self.n = datasets.shape
        # 弱分类器集合
        self.clf_sets = []
        # 初始化weights
        self.weights = [1.0/self.m]*self.m
        # G(x)系数
        self.alpha = []

    # def _G(self, features, labels, weights):
    #     v = [weights[i]*features[i] for i in range(self.m)]
    #     error = 0
    #     for i in range(self.m):
    #         if (features[i] > v and y[i] == -1) or (features[i] < v and y[i] == 1):
    #             error += 1
    #     return error/self.m
    def _G(self, feature, v):
        if feature > v:
            return 1
        else:
            return -1

    def _alpha(self, em):
        return 0.5*np.log((1-em)/em)

    def _em(self,features, labels, weights):
        v = [weights[i] * features[i] for i in range(self.m)]
        error = 0
        for i in range(self.m):
            if self._G(features[i], v) != y[i]:
                error += 1
        return error/self.m, v
    def _Z(self,m,v):
        return sum([self.weights[i]*np.exp(-self.alpha[m]*self.y[i]*self._G(self.X[i],v)) for i in range(self.m)])
    def train(self,X,y):
        self.init_arg(X,y)
        em = np.zeros(self.n)
        v = 0
        for epoch in range(self.n):
            em[epoch],v = self._em(self.X[:,epoch], self.y, self.weights)

        alpha = self._alpha(min(em))
        for i in range(self.n):
            self.weights[i] = (self.weights[i]/self._Z(i,v))*np.exp(-self.alpha[i])


if __name__ == '__main__':
    X,y = create_data()
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
    boost = Boost()
    boost.train(X_train, y_train)
