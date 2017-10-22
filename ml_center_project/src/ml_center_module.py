"""
Created on October 18, 2017

Machine learning and another center (other than the analytic center)

"""
__author__ = 'amm'
__date__ = "Oct 18, 2017"
__version__ = 0.0

import numpy as np
from scipy.optimize import linprog as lp
from sklearn import svm, datasets
from sklearn.metrics.pairwise import pairwise_kernels
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=100, edgeitems='all', suppress=True,
                    precision=2)

class ThisSVC(object):
    """
    Support vector classifier to compare.
    """
    def __init__(self, svc_C, svc_kernel, svc_degree, svc_gamma, svc_coef0, \
                 svc_cache_size):
        """
        SVM classification parameters
        --------------
        svc_C           : 10000 to have no data points outside the tube
        svc_kernel      : 'poly', 'rbf'
        svc_degree      : (u'v + svc_coef0)^svc_degree, ignored by rbf and other kernels
        svc_gamma       : exp(svc_gamma * ||u - v||)
                          svc_gamma = 1/2sigma^2
        svc_coef0       : (u'v + svc_coef0)^svc_degree, poly and sigmoid kernel only
        svc_cache_size  : 200
        svc_tol         : default 1e-3
        svc_verbose     : default False
        svc_max_iter    : default -1 for not limit
        """
        self.svc_C = svc_C
        self.svc_kernel = svc_kernel
        self.svc_degree = svc_degree
        self.svc_gamma = svc_gamma
        self.svc_coef0 = svc_coef0
        self.svc_cache_size = svc_cache_size
        self.svc_tol = svc_tol
        self.svc_verbose = svc_verbose
        self.svc_max_iter = svc_max_iter

    def get_data(self, trX, trY, tsX, tsY):
        """
        Dataset
        -------
        trX: input training data
        trY: label training data
        tsX: input test data
        tsY: label test data
        """
        self.trX = trX
        self.trY = trY
        self.tsX = tsX
        self.tsY = tsY

    def run_this_svc(self):
        """
        SVM Classification and Prediction
        ---------------------------------
        """

        clf = svm.SVC(C=self.svc_C, cache_size=self.svc_cache_size, \
                      coef0=self.svc_coef0, degree=self.svc_degree, \
                      gamma=self.svc_gamma, kernel=self.svc_kernel, \
                      max_iter=self.svc_max_iter, tol=self.svc_tol, \
                      verbose=self.svc_verbose)
        clf.fit(self.trX, self.trY)
        predY = clf.predict(self.tsX)
        self.predY = predY

class CenterMachine(object):
    """
    New learning machine
    """
    def __init__(self):
        pass

def get_label_adjusted_kernel(trainx, trainy):
    """
    Compute the kernel matrix which takes labels into consideration.  The kernel
    matrix has l rows and l+1 columns, where l is the number of samples in trainx
    and trainy. This function returns K.T from Trafalis, Malyscheff, ACM, 2002.

    Parameters
    ----------
    In    : trainx, input samples (d by l)
            trainy, labels (d by 1)
    Out   : kmat

    Usage:
    ------
    kmat = get_label_adjusted_kernel(trainx, trainy)
    """
    k = pairwise_kernels(X=trX, metric='linear')
    # multiply by labels and add row, same K as in Trafalis Malyscheff, ACM, 2002
    K = np.array([k[i, :] * trY for i in range(len(k[0, :]))])
    K = np.vstack((K, trY))
    return K.T

def run_ml_center_learner(trainx, trainy):
    """
    Build linear learning model by minimizing the epigraph of hyperplanes in
    kernelized version space.

    Parameters
    ----------
    In    : trainx, input samples (d by l)
            trainy, labels (d by 1)
    Out   : res, optimal weights (alpha_1, ..., alpha_l, b) and optimal eps

    Usage:
    ------
    res = run_ml_center_learner(trainx, trainy)
    """
    [num_samples, num_features] = trainx.shape

    # Objective function
    c = np.vstack((np.zeros((num_samples+1, 1)), 1)).flatten()

    # Constraints from data (halfspaces)
    kmat = get_label_adjusted_kernel(trainx, trainy)
    Aub_data = np.hstack((-kmat, -np.ones((num_samples, 1))))
    bub_data = np.zeros((num_samples, 1))

    # Box constraints lower
    Aub_box_lower = np.hstack((-np.identity(num_samples+1), -np.ones((num_samples+1, 1))))
    bub_box_lower = np.ones((num_samples+1, 1))

    # Box constraints upper
    Aub_box_upper = np.hstack((np.identity(num_samples+1), -np.ones((num_samples+1, 1))))
    bub_box_upper = np.ones((num_samples+1, 1))

    # Putting it all together
    Aub = np.vstack((Aub_data, Aub_box_lower, Aub_box_upper))
    bub = np.vstack((bub_data, bub_box_lower, bub_box_upper)).flatten()

    # TODO: needed?
    x0_bounds = (None, None)
    x1_bounds = (None, None)
    x2_bounds = (None, None)
    x3_bounds = (None, None)
    x4_bounds = (None, None)
    x5_bounds = (None, None)
    res = lp(c=c, A_ub=Aub, b_ub=bub, bounds=(x0_bounds, x1_bounds, x2_bounds, x3_bounds, x4_bounds, x5_bounds))
    return res


def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

if __name__ == '__main__':
    """
    execfile('ml_center_module.py')
    springer thank you: thankyou1710
    """
    import os

    os.chdir('C:\\Users\\amalysch\\PycharmProjects\\ml_center_repository\\ml_center_project\\src')

    svc_C = 10000
    svc_kernel = 'rbf'
    svc_degree = 3  # ignored by rbf and other kernels
    svc_gamma = 1.0
    svc_coef0 = 1.0  # poly and sigmoid kernel only
    svc_cache_size = 500
    svc_tol = 0.001  # default 1e-3
    svc_verbose = False
    svc_max_iter = -1  # default -1 for not limit

    # trX = [[0, 0], [1, 1]]
    # trY = [0, 1]
    # tsX = [[2., 2.]]
    # tsY = [1]
    # Total sample size 150
    iris = datasets.load_iris()
    X = iris.data[:, :2]
    Y = iris.target
    # Samples 0...99 contain only class 0 and 1
    X = X[:100, :]
    Y = Y[:100]
    trX = np.vstack((X[:25, :], X[75:, :]))
    tsX = X[25:75, :]
    trY = np.hstack((Y[:25], Y[75:]))
    tsY = Y[25:75]

    C = 1.0  # SVM regularization parameter
    models = (svm.SVC(kernel='linear', C=C),
              svm.LinearSVC(C=C),
              svm.SVC(kernel='rbf', gamma=7.7, C=C),
              svm.SVC(kernel='poly', degree=3, C=C))
    models = (clf.fit(trX, trY) for clf in models)

    # title for the plots
    titles = ('SVC with linear kernel',
              'LinearSVC (linear kernel)',
              'SVC with RBF kernel',
              'SVC with polynomial (degree 3) kernel')

    # Set-up 2x2 grid for plotting.
    fig, sub = plt.subplots(2, 2)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    X0, X1 = trX[:, 0], trX[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    for clf, title, ax in zip(models, titles, sub.flatten()):
        plot_contours(ax, clf, xx, yy,
                      cmap=plt.cm.coolwarm, alpha=0.8)
        ax.scatter(X0, X1, c=trY, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xlabel('Sepal length')
        ax.set_ylabel('Sepal width')
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)

    plt.show()

    # Loading AND data
    EPSTOL = 1e-6
    trX = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
    trY = np.array([1, -1, -1, -1])

    # Fast nonkernelized LP
    [num_samples, num_features] = trX.shape
    c = np.vstack((np.zeros((num_features+1, 1)), 1)).flatten()

    # Change 0 labels to -1
    trY = [i if i == 1 else -1 for i in trY]
    # Include bias b
    trX = np.hstack((trX, np.ones((num_samples, 1))))
    k = np.multiply(np.array(trY).reshape(len(trY), 1), trX)
    Aub_data = np.hstack((-k, -np.ones((num_samples, 1))))
    bub_data = np.zeros((num_samples, 1))

    # num_features+1 to account for bias
    d1 = -np.identity(num_features+1)
    d2 = np.identity(num_features+1)
    kbox = np.vstack((d1, d2))
    Aub_box = np.hstack((kbox, -np.ones((2*(num_features+1), 1))))
    bub_box = np.ones((2*(num_features+1), 1))

    Aub = np.vstack((Aub_data, Aub_box))
    bub = np.vstack((bub_data, bub_box)).flatten()
    x0_bounds = (None, None)
    x1_bounds = (None, None)
    x2_bounds = (None, None)
    x3_bounds = (None, None)
    # x0_bounds = (-10, 10)
    # x1_bounds = (-10, 10)
    # x2_bounds = (-10, 10)
    # x3_bounds = (-10, 10)
    res = lp(c=c, A_ub=Aub, b_ub=bub, bounds=(x0_bounds, x1_bounds, x2_bounds, x3_bounds))
    # res = lp(c=c, A_ub=Aub, b_ub=bub)

    res.x
    wopt = res.x
    print "wopt = ", wopt
    epsopt = res.fun
    print "epsopt = ", epsopt
    print "Feasibility check: \n", np.dot(Aub, wopt) < bub + EPSTOL


    # this_svc = ThisSVC(svc_C, svc_kernel, svc_degree, svc_gamma, svc_coef0, svc_cache_size)
    # this_svc.get_data(trX, trY, tsX, tsY)
    # this_svc.run_this_svc()



