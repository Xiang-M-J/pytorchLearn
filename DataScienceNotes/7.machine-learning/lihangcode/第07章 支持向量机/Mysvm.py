import sys

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from os import error


def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    # print(df)
    df['label'] = iris.target
    df.columns = [
        'sepal length', 'sepal width', 'petal length', 'petal width', 'label'
    ]
    data = np.array(df.iloc[:100, [0, 1, -1]])  # 取sepal length和 sepal width两列和label列进行分析
    for i in range(len(data)):
        if data[i, -1] == 0:
            data[i, -1] = -1
    return data[:, :2], data[:, -1]


class SVM:
    def __init__(self, max_iter, kernel='linear', theta=1):
        self.max_iter = max_iter
        self.kernel = kernel
        self.theta = theta

    def init_args(self, features, label):
        self.m, self.n = features.shape
        self.X = features
        self.y = label
        self.C = 1.0  # 松弛参数
        self.b = 0
        self.alpha = np.zeros(self.m)  # α
        self.E = [self._g(i) - self.y[i] for i in range(self.m)]

    def _g(self, i):
        r = self.b
        for j in range(self.m):
            r += self.alpha[j] * self.y[i] * self.Kernel(self.X[i], self.X[j])
        return r

    def initalpha(self):
        index_list = [i for i in range(self.m) if 0 < self.alpha[i] < self.C]
        # 否则遍历整个训练集
        non_satisfy_list = [i for i in range(self.m) if i not in index_list]
        index_list.extend(non_satisfy_list)
        # 具体参见 P129
        for i in index_list:
            if self.isKKT(i):  # 符合KKT条件的样本点跳过
                continue

            E1 = self.E[i]  # 第一个不符合KKT条件作为第一个参数
            # 如果E2是+，选择最小的；如果E2是负的，选择最大的
            if E1 >= 0:
                j = min(range(self.m), key=lambda x: self.E[x])
            else:
                j = max(range(self.m), key=lambda x: self.E[x])
            return i, j

    # 该条件具体参见P128-P129 (7.111, 7.112, 7.113)
    def isKKT(self, i):
        y_g = self._g(i) * self.y[i]
        if self.alpha[i] == 0:
            return y_g >= 1
        elif 0 < self.alpha[i] < self.C:
            return y_g == 1
        else:
            return y_g <= 1

    def Kernel(self, x1, x2):
        if self.kernel == 'linear':
            return sum([x1[k] * x2[k] for k in range(self.n)])
        if self.kernel == 'poly':
            return (sum([x1[k] * x2[k] for k in range(self.n)]) + 1) ** 3
        if self.kernel == 'Gauss':
            return np.exp(-(sum((x1[k] - x2[k]) ** 2 for k in range(self.n))) / (2 * self.theta * self.theta))
        print("未定义kernel")
        return 0

    def train(self, features, labels):
        self.init_args(features, labels)
        for iter in range(self.max_iter):
            print(f"第{iter + 1}次迭代")
            i1, i2 = self.initalpha()  # a1, a2两个参数
            if self.y[i1] == self.y[i2]:  # 计算L和H 具体参见P126
                L = max(0, self.alpha[i2] + self.alpha[i1] - self.C)
                H = min(self.C, self.alpha[i2] + self.alpha[i1])
            else:
                L = max(0, self.alpha[i2] - self.alpha[i1])
                H = min(self.C, self.C + self.alpha[i2] - self.alpha[i1])

            # 以下内容参考 定理7.6 P127
            eta = self.Kernel(self.X[i1], self.X[i1]) + self.Kernel(self.X[i2], self.X[i2]) \
                  - 2 * self.Kernel(self.X[i1], self.X[i2])  # 求解η，η=K11+K22-2K12

            try:
                alpha2_new_unc = self.alpha[i2] + self.y[i2] * (self.E[i1] - self.E[i2]) / eta
            except ZeroDivisionError:
                alpha2_new_unc = sys.maxsize                # sys.maxsize = 9223372036854775807
                pass

            if alpha2_new_unc > H:
                alpha2_new = H
            elif L <= alpha2_new_unc <= H:
                alpha2_new = alpha2_new_unc
            else:
                alpha2_new = L
            alpha1_new = self.alpha[i1] + self.y[i1] * self.y[i2] * (self.alpha[i2] - alpha2_new)

            # 以下内容参见P129 计算阈值b和插值Ei
            b1 = -self.E[i1] - self.y[i1] * self.Kernel(self.X[i1], self.X[i1]) * (alpha1_new - self.alpha[i1]) \
                 - self.y[i2] * self.Kernel(self.X[i2], self.X[i1]) * (alpha2_new - self.alpha[i2]) + self.b

            b2 = -self.E[i2] - self.y[i1] * self.Kernel(self.X[i1], self.X[i2]) * (alpha1_new - self.alpha[i1]) \
                 - self.y[i2] * self.Kernel(self.X[i2], self.X[i2]) * (alpha2_new - self.alpha[i2]) + self.b

            if 0 < alpha1_new < self.C:
                b_new = b1
            elif 0 < alpha2_new < self.C:
                b_new = b2
            else:
                # 选择中点
                b_new = (b1 + b2) / 2
            self.alpha[i1] = alpha1_new
            self.alpha[i2] = alpha2_new
            self.b = b_new
            self.E[i1] = self._g(i1) - self.y[i1]
            self.E[i2] = self._g(i2) - self.y[i2]
        print("train done")

    def predict(self, data):
        r = self.b
        for i in range(self.m):
            r += self.alpha[i] * self.y[i] * self.Kernel(data, self.X[i])

        return 1 if r > 0 else -1

    def score(self, X_test, y_test):
        right_count = 0
        for i in range(len(X_test)):
            result = self.predict(X_test[i])
            if result == y_test[i]:
                right_count += 1
        return right_count / len(X_test)


if __name__ == '__main__':
    X, y = create_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    svm = SVM(max_iter=100, kernel='Gauss', theta=2)
    svm.train(X_train, y_train)
    score = svm.score(X_test, y_test)
    print(score)
    # plt.scatter(X[:50, 0], X[:50, 1], label='0')
    # plt.scatter(X[50:, 0], X[50:, 1], label='1')
    # plt.legend()
    # plt.show()
