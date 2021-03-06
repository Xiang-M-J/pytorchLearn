#!/usr/bin/env python
# coding: utf-8

# # 2-推荐系统

# In[1]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context="notebook", style="white", palette=sns.color_palette("RdBu"))
import numpy as np
import pandas as pd
import scipy.io as sio


# # load data and setting up
# % Notes: X - num_movies (1682)  x num_features (10) matrix of movie features
# %        Theta - num_users (943)  x num_features (10) matrix of user features
# %        Y - num_movies x num_users matrix of user ratings of movies
# %        R - num_movies x num_users matrix, where R(i, j) = 1 if the
# %            i-th movie was rated by the j-th user
# In[2]:


movies_mat = sio.loadmat('./data/ex8_movies.mat')
Y, R = movies_mat.get('Y'), movies_mat.get('R')

print(Y.shape, R.shape)


# In[3]:


m, u = Y.shape
# m: how many movies
# u: how many users

n = 10  # how many features for a movie


# In[4]:


param_mat = sio.loadmat('./data/ex8_movieParams.mat')
theta, X = param_mat.get('Theta'), param_mat.get('X')

print(theta.shape, X.shape)


# # cost
# ![](data:image/png;base64,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)

# In[5]:


def serialize(X, theta):
    """serialize 2 matrix
    """
    # X (movie, feature), (1682, 10): movie features
    # theta (user, feature), (943, 10): user preference
    return np.concatenate((X.ravel(), theta.ravel()))


def deserialize(param, n_movie, n_user, n_features):
    """into ndarray of X(1682, 10), theta(943, 10)"""
    return param[:n_movie * n_features].reshape(n_movie, n_features),            param[n_movie * n_features:].reshape(n_user, n_features)


# recommendation fn
def cost(param, Y, R, n_features):
    """compute cost for every r(i, j)=1
    Args:
        param: serialized X, theta
        Y (movie, user), (1682, 943): (movie, user) rating
        R (movie, user), (1682, 943): (movie, user) has rating
    """
    # theta (user, feature), (943, 10): user preference
    # X (movie, feature), (1682, 10): movie features
    n_movie, n_user = Y.shape
    X, theta = deserialize(param, n_movie, n_user, n_features)

    inner = np.multiply(X @ theta.T - Y, R)

    return np.power(inner, 2).sum() / 2


def gradient(param, Y, R, n_features):
    # theta (user, feature), (943, 10): user preference
    # X (movie, feature), (1682, 10): movie features
    n_movies, n_user = Y.shape
    X, theta = deserialize(param, n_movies, n_user, n_features)

    inner = np.multiply(X @ theta.T - Y, R)  # (1682, 943)

    # X_grad (1682, 10)
    X_grad = inner @ theta

    # theta_grad (943, 10)
    theta_grad = inner.T @ X

    # roll them together and return
    return serialize(X_grad, theta_grad)


def regularized_cost(param, Y, R, n_features, l=1):
    reg_term = np.power(param, 2).sum() * (l / 2)

    return cost(param, Y, R, n_features) + reg_term


def regularized_gradient(param, Y, R, n_features, l=1):
    grad = gradient(param, Y, R, n_features)
    reg_term = l * param

    return grad + reg_term


# In[7]:


# use subset of data to calculate the cost as in pdf...
users = 4
movies = 5
features = 3

X_sub = X[:movies, :features]
theta_sub = theta[:users, :features]
Y_sub = Y[:movies, :users]
R_sub = R[:movies, :users]

param_sub = serialize(X_sub, theta_sub)
cost(param_sub, Y_sub, R_sub, features)


# In[9]:


param = serialize(X, theta)  # total real params

cost(serialize(X, theta), Y, R, 10)  # this is real total cost


# # gradient
# <img style="float: left;" src="../img/rcmd_gradient.png">

# In[11]:


n_movie, n_user = Y.shape

X_grad, theta_grad = deserialize(gradient(param, Y, R, 10),
                                      n_movie, n_user, 10)


# <img style="float: left;" src="../img/rcmd_vectorized_grad.png">

# In[30]:


assert X_grad.shape == X.shape
assert theta_grad.shape == theta.shape


# # regularized cost

# In[12]:


# in the ex8_confi.m, lambda = 1.5, and it's using sub data set
regularized_cost(param_sub, Y_sub, R_sub, features, l=1.5)


# In[13]:


regularized_cost(param, Y, R, 10, l=1)  # total regularized cost


# # regularized gradient

# <img style="float: left;" src="../img/rcmd_reg_grad.png">

# In[14]:


n_movie, n_user = Y.shape

X_grad, theta_grad = deserialize(regularized_gradient(param, Y, R, 10),
                                                                n_movie, n_user, 10)

assert X_grad.shape == X.shape
assert theta_grad.shape == theta.shape


# # parse `movie_id.txt`

# In[15]:


movie_list = []

with open('./data/movie_ids.txt', encoding='latin-1') as f:
    for line in f:
        tokens = line.strip().split(' ')
        movie_list.append(' '.join(tokens[1:]))

movie_list = np.array(movie_list)


# # reproduce my ratings

# In[16]:


ratings = np.zeros(1682)

ratings[0] = 4
ratings[6] = 3
ratings[11] = 5
ratings[53] = 4
ratings[63] = 5
ratings[65] = 3
ratings[68] = 5
ratings[97] = 2
ratings[182] = 4
ratings[225] = 5
ratings[354] = 5


# # prepare data

# In[17]:


Y, R = movies_mat.get('Y'), movies_mat.get('R')


Y = np.insert(Y, 0, ratings, axis=1)  # now I become user 0
print(Y.shape)


# In[18]:


R = np.insert(R, 0, ratings != 0, axis=1)
print(R.shape)


# In[19]:


n_features = 50
n_movie, n_user = Y.shape
l = 10


# In[20]:


X = np.random.standard_normal((n_movie, n_features))
theta = np.random.standard_normal((n_user, n_features))

print(X.shape, theta.shape)


# In[21]:


param = serialize(X, theta)


# normalized ratings

# In[22]:


Y_norm = Y - Y.mean()
Y_norm.mean()


# # training

# In[23]:


import scipy.optimize as opt


# In[26]:


res = opt.minimize(fun=regularized_cost,
                   x0=param,
                   args=(Y_norm, R, n_features, l),
                   method='TNC',
                   jac=regularized_gradient)
#这里很慢


# In[27]:


print(res)


# In[29]:


X_trained, theta_trained = deserialize(res.x, n_movie, n_user, n_features)
print(X_trained.shape, theta_trained.shape)


# In[30]:


prediction = X_trained @ theta_trained.T


# In[31]:


my_preds = prediction[:, 0] + Y.mean()


# In[32]:


idx = np.argsort(my_preds)[::-1]  # Descending order
print(idx.shape)


# In[33]:


# top ten idx
print(my_preds[idx][:10])


# In[34]:


for m in movie_list[idx][:10]:
    print(m)


# In[ ]:




