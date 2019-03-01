#!/usr/bin/env python

import random
import numpy as np
from sklearn.metrics import log_loss
from numpy.linalg import norm

from config import *

"""
calculate regularized cross-entropy loss
input:
    @theta: parameter matrix
    @X: sample matrix
    @Y: label vector
    @C: regularization coeff
output:
    corss-entropy loss + 0.5 * (1 / C) * norm(theta)
"""
def entropy_loss(theta, X, Y, C = 0.1):
    return C * log_loss(Y, predict_proba(theta, X), normalize = False) + 0.5 * norm(theta) ** 2;

"""
predict label of samples
input:
    @theta: parameter matrix
    @X: sample matrix
output:
    label vector
"""
def predict(theta, X):
    wx = np.matrix(X.dot(theta));
    return np.array(np.argmax(wx, axis = 1).transpose().astype(int))[0] + 1;

"""
calculate probabiliry of each sample as each class
input:
    @theta: parameter matrix
    @X: sample matrix
output:
    probabiliry matrix
"""
def predict_proba(theta, X):
    wx = np.matrix(X.dot(theta));
    shift_wx = wx - wx.max(axis = 1).dot(np.ones((1, len(theta[0]))));
    exp_wx = np.exp(shift_wx);
    return exp_wx / exp_wx.sum(axis = 1);

"""
a helper function to convert 1-d index to 2-d position
"""
def idx2pos(theta, d):
    return d % len(theta), d / len(theta);

"""
calculate partial derivative of loss function to d-th parameter theta[d]
    @theta: parameter matrix
    @d: index of the parameter
    @X: training sample matrix
    @Y: training label vector
    @h: a small amount to estimate derivative, default: eps
"""
def get_d1_loss(theta, d, X, Y, h = eps):
    loss = entropy_loss(theta, X, Y);
    x, y = idx2pos(theta, d);
    origin_val = theta[x][y];
    theta[x][y] = origin_val + h;
    diff = (entropy_loss(theta, X, Y) - loss) / h;
    theta[x][y] = origin_val;
    return diff;

"""
calculate second-order partial derivative of loss function to d-th parameter theta[d]
    @theta: parameter matrix
    @d: index of the parameter
    @X: training sample matrix
    @Y: training label vector
    @h: a small amount to estimate derivative, default: eps
"""
def get_d2_loss(theta, d, X, Y, h = eps):
    x, y = idx2pos(theta, d);
    origin_val = theta[x][y];
    d2loss = (get_d1_loss(theta, d, X, Y, h) - get_d1_loss(theta, d, X, Y, -h)) / h;
    theta[x][y] = origin_val;
    return d2loss;

"""
update corresponding parameter with newton method
input:
    @theta: parameter matrix
    @d: index of the parameter
    @X: training sample matrix
    @Y: training label vector
    @eps: judge whether converge
output:

"""
def update_coordinate(theta, d, X, Y, eps = 0.00001):
    loss = entropy_loss(theta, X, Y);
    x, y = idx2pos(theta, d);
    origin_val = theta[x][y];
    diff = get_d1_loss(theta, d, X, Y);
    d2loss = get_d2_loss(theta, d, X, Y);
    if abs(d2loss) > 0.0:
        theta[x][y] -= diff / d2loss;
    new_loss = entropy_loss(theta, X, Y);
    return abs(new_loss - loss) >= eps;
