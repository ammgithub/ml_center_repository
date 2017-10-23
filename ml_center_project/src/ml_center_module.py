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


class FastKernelMachine(object):
    """
    A very fast kernel machine

    Parameters
    ----------
    kernel : string, 'linear', 'poly', 'rbf' (default='linear')
    Describes the kernel function.

    degree : int, (default=3)
    Degree of the polynomial kernel. Ignored by other kernels.

    gamma : int, (default=1)
    For radial-basis function kernels. gamma = 1 / 2 sigma^2.

    coef0 : float, (default=0.0)
    For poly and sigmoid kernels only.

    Examples
    --------
    >>> fkm = FastKernelMachine()
    >>> fkm = FastKernelMachine(kernel='poly', degree=3, gamma=1, coef0=1)

    Notes
    -----

    References
    ----------
    """
    def __init__(self, kernel='linear', degree=3, gamma=1, coef0=0.0,
                 verbose=False):
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0

    def fit(self, trainx, trainy):
        """
        Compute the optimal weight vector to classify (trainx, trainy).

        Parameters
        ----------
        trainx : numpy array of floats, num_samples-by-num_features
                 Input training samples

        trainy : list of numpy array of floats or integers num_samples-by-one
                 Input training labels

        Returns
        -------
        self : object

        """
        [self.num_samples, self.num_features] = trainx.shape

        # Need trainx and testx in 'predict', need trainy in 'plot2d'
        self.trainx = trainx
        self.trainy = trainy

        # Objective function
        c = np.vstack((np.zeros((self.num_samples+1, 1)), 1)).flatten()

        # Constraints from data (halfspaces)
        kmat = get_label_adjusted_train_kernel(trainx, trainy)
        Aub_data = np.hstack((-kmat, -np.ones((self.num_samples, 1))))
        bub_data = np.zeros((self.num_samples, 1))

        # Box constraints lower
        Aub_box_lower = np.hstack((-np.identity(self.num_samples+1),
                                   -np.ones((self.num_samples+1, 1))))
        bub_box_lower = np.ones((self.num_samples+1, 1))

        # Box constraints upper
        Aub_box_upper = np.hstack((np.identity(self.num_samples+1),
                                   -np.ones((self.num_samples+1, 1))))
        bub_box_upper = np.ones((self.num_samples+1, 1))

        # Putting it all together
        Aub = np.vstack((Aub_data, Aub_box_lower, Aub_box_upper))
        bub = np.vstack((bub_data, bub_box_lower, bub_box_upper)).flatten()
        res = lp(c=c, A_ub=Aub, b_ub=bub, bounds=(None, None))
        weight_opt = res.x

        # last element is epsilon
        self.weight_opt = weight_opt[:-1]
        self.eps_opt = res.fun

    def predict(self, testx):
        """
        Predict functional values of testx using weight_opt computed in 'fit'.

        Parameters
        ----------
        testx :  numpy array of floats, num_samples-by-num_features
                 Input test samples

        Returns
        -------
        self : object
        """
        ktest = get_label_adjusted_test_kernel(self.trainx, testx)

        # want printed output on console
        return np.sign(np.dot(ktest, self.weight_opt))

    def plot2d(self, meshstep=0.02):
        """
        Plot simple examples that have 2-dimensional input training samples
        (2 features)

        Parameters
        ----------
        meshstep : float, (default=0.02)
                   Precision in meshgrid, smaller values result in smoother functions.

        Returns
        -------
        self : object
        """
        # TODO: pass clf object directly to plot_countours
        if self.trainx.shape[1] == 2:
            xmin = self.trainx[:, 0].min() - 1
            xmax = self.trainx[:, 0].max() + 1
            ymin = self.trainx[:, 1].min() - 1
            ymax = self.trainx[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(xmin, xmax + meshstep, meshstep),
                                 np.arange(ymin, ymax + meshstep, meshstep))
            # TODO: Want this as: Z = predict_ml_center(np.c_[xx.ravel(), yy.ravel()])

            fig, ax = plt.subplots(1, 1)
            testx = np.c_[xx.ravel(), yy.ravel()]
            Z = self.predict(testx)

            # Z = predict_ml_center(weight_opt, trainx, np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            out = ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
            ax.scatter(self.trainx[:, 0], self.trainx[:, 1], c=self.trainy,
                       cmap=plt.cm.coolwarm, s=60, edgecolors='k')
            plt.show()
        else:
            return "Input sample dimension must be equal to 2. Exiting. "


def get_label_adjusted_train_kernel(trainx, trainy):
    """
    Compute the training kernel matrix. This matrix also takes labels into consideration.
    The training kernel matrix has l+1 rows and l columns, where l is the number of
    samples in trainx and trainy.
    All columns are multiplied by the corresponding yj.  This implies that the l+1 row
    contains yj's.  Corresponds to K.T from Trafalis, Malyscheff, ACM, 2002.

    Parameters
    ----------
    In    : trainx, input samples for training set (d by l)
            trainy, labels for training set (d by 1) (flattened)
    Out   : ktrain

    Usage:
    ------
    ktrain = get_label_adjusted_train_kernel(trainx, trainy)
    """
    # k = pairwise_kernels(X=trX, metric='linear')
    k = pairwise_kernels(X=trainx, metric='poly', degree=3, coef0=1)
    # multiply by labels and add row, same K as in Trafalis Malyscheff, ACM, 2002
    K = np.array([k[i, :] * trY for i in range(len(k[0, :]))])
    K = np.vstack((K, trY))
    return K.T


def get_label_adjusted_test_kernel(trainx, testx):
    """
    Compute the test kernel matrix.  The test kernel matrix has l+1 rows and l columns,
    however, the l+1 row has 1s, not the labels yj. Columns are not multiplied by yj.

    Parameters
    ----------
    In    : trainx, input samples for training set (d by l)
            testx, input samples for test set (d by num_test_samples)
    Out   : ktrain

    Usage:
    ------
    ktest = get_label_adjusted_test_kernel(trainx, testx)
    """
    num_test_samples = testx.shape[0]
    # ktest = pairwise_kernels(X=trainx, Y=testx, metric='linear')
    ktest = pairwise_kernels(X=trainx, Y=testx,  metric='poly', degree=3, coef0=1)
    # add row of ones
    Ktest = np.vstack((ktest, np.ones((1, num_test_samples))))
    return Ktest.T


def learn_ml_center(trainx, trainy):
    """
    Build linear learning model by minimizing the epigraph of hyperplanes in
    kernelized version space.

    Parameters
    ----------
    In    : trainx, input samples (l by d)
            trainy, labels (l) (flattened)
    Out   : weight_opt, eps_opt
            optimal weights (alpha_1, ..., alpha_l, b) and optimal eps

    Usage:
    ------
    weight_opt, eps_opt = learn_ml_center(trainx, trainy)
    """
    [num_samples, num_features] = trainx.shape

    # Objective function
    c = np.vstack((np.zeros((num_samples+1, 1)), 1)).flatten()

    # Constraints from data (halfspaces)
    kmat = get_label_adjusted_train_kernel(trainx, trainy)
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
    res = lp(c=c, A_ub=Aub, b_ub=bub, bounds=(None, None))
    weight_opt = res.x
    # last element is epsilon
    weight_opt = weight_opt[:-1]
    eps_opt = res.fun
    return weight_opt, eps_opt


def predict_ml_center(weight_opt, trainx, testx):
    """
    Predict test set using ml_center model. weight_opt is the optimal vector of weights.
    trainx is the training input.  testx is the test input.

    Parameters
    ----------
    In    : weight_opt, trainx, testx
    Out   : ftestx

    Usage:
    ------
    ftestx = predict_ml_center(weight_opt, trainx, testx)
    """
    ktest = get_label_adjusted_test_kernel(trainx, testx)
    return np.sign(np.dot(ktest, weight_opt))


def plot_data_and_contours(ax, trainx, trainy, meshstep=0.02):
    """
    Build linear learning model by minimizing the epigraph of hyperplanes in
    kernelized version space.

    Parameters
    ----------
    In    : ax object
          : trainx, input samples (l by d)
            trainy, labels (l) (flattened)
    Out   : out

    Usage:
    ------
    plot_data_and_contours(ax, trainx, trainy, meshstep=0.02)
    """
    # TODO: pass clf object directly to plot_countours
    if trainx.shape[1] == 2:
        xmin = trainx[:, 0].min() - 1
        xmax = trainx[:, 0].max() + 1
        ymin = trainx[:, 1].min() - 1
        ymax = trainx[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(xmin, xmax+meshstep, meshstep),
                             np.arange(ymin, ymax+meshstep, meshstep))
        # TODO: Want this as: Z = predict_ml_center(np.c_[xx.ravel(), yy.ravel()])
        Z = predict_ml_center(weight_opt, trainx, np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        out = ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        ax.scatter(trX[:, 0], trX[:, 1], c=trY, cmap=plt.cm.coolwarm, s=60, edgecolors='k')
        return out
    else:
        return "Input sample dimension must be equal to 2. Exiting. "


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
    z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    out = ax.contourf(xx, yy, z, **params)
    return out

if __name__ == '__main__':
    """
    execfile('ml_center_module.py')
    springer thank you: thankyou1710
    """
    import os

    os.chdir('C:\\Users\\amalysch\\PycharmProjects\\ml_center_repository\\ml_center_project\\src')

    # Testing AND data
    trX = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
    trY = [1, -1, -1, -1]
    tsX = np.array([[1, 2], [-3, 2], [6, -1]])
    tsY = [1, -1, 1]
    weight_opt, eps_opt = learn_ml_center(trX, trY)
    ftestx = predict_ml_center(weight_opt, trX, tsX)
    print "weight_opt = ", weight_opt
    print "eps_opt = ", eps_opt
    print "[ftestx, tsY] = \n", np.array([ftestx, np.array(tsY)]).T

    fig, ax = plt.subplots(1, 1)
    # plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    # ax = sub.flatten()
    out = plot_data_and_contours(ax, trX, trY, meshstep=0.02)

    # Testing OR data
    trX = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
    trY = [1, -1, 1, -1]
    tsX = np.array([[1, 2], [-3, 2], [6, -1]])
    tsY = [1, -1, 1]
    weight_opt, eps_opt = learn_ml_center(trX, trY)
    ftestx = predict_ml_center(weight_opt, trX, tsX)
    print "weight_opt = ", weight_opt
    print "eps_opt = ", eps_opt
    print "[ftestx, tsY] = \n", np.array([ftestx, np.array(tsY)]).T

    fig, ax = plt.subplots(1, 1)
    # plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    # ax = sub.flatten()
    out = plot_data_and_contours(ax, trX, trY, meshstep=0.02)

    plt.show()

    print "\n Now let's do that again with an object... \n"
    # Testing OR data
    trX = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
    trY = [1, -1, 1, -1]
    tsX = np.array([[1, 2], [-3, 2], [6, -1]])
    tsY = [1, -1, 1]
    fkm = FastKernelMachine(kernel='poly', degree=3, gamma=1, coef0=1)
    fkm.fit(trX, trY)
    fkm.predict(tsX)
    fkm.plot2d()

    # svc_C = 10000
    # svc_kernel = 'rbf'
    # svc_degree = 3  # ignored by rbf and other kernels
    # svc_gamma = 1.0
    # svc_coef0 = 1.0  # poly and sigmoid kernel only
    # svc_cache_size = 500
    # svc_tol = 0.001  # default 1e-3
    # svc_verbose = False
    # svc_max_iter = -1  # default -1 for not limit
    #
    # iris = datasets.load_iris()
    # X = iris.data[:, :2]
    # Y = iris.target
    # # Samples 0...99 contain only class 0 and 1
    # X = X[:100, :]
    # Y = Y[:100]
    # trX = np.vstack((X[:25, :], X[75:, :]))
    # tsX = X[25:75, :]
    # trY = np.hstack((Y[:25], Y[75:]))
    # tsY = Y[25:75]
    #
    # C = 1.0  # SVM regularization parameter
    # models = (svm.SVC(kernel='linear', C=C),
    #           svm.LinearSVC(C=C),
    #           svm.SVC(kernel='rbf', gamma=7.7, C=C),
    #           svm.SVC(kernel='poly', degree=3, C=C))
    # models = (clf.fit(trX, trY) for clf in models)
    #
    # # title for the plots
    # titles = ('SVC with linear kernel',
    #           'LinearSVC (linear kernel)',
    #           'SVC with RBF kernel',
    #           'SVC with polynomial (degree 3) kernel')
    #
    # # Set-up 2x2 grid for plotting.
    # fig, sub = plt.subplots(2, 2)
    # plt.subplots_adjust(wspace=0.4, hspace=0.4)
    #
    # X0, X1 = trX[:, 0], trX[:, 1]
    # xx, yy = make_meshgrid(X0, X1)
    #
    # for clf, title, ax in zip(models, titles, sub.flatten()):
    #     plot_contours(ax, clf, xx, yy,
    #                   cmap=plt.cm.coolwarm, alpha=0.8)
    #     ax.scatter(X0, X1, c=trY, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    #     ax.set_xlim(xx.min(), xx.max())
    #     ax.set_ylim(yy.min(), yy.max())
    #     ax.set_xlabel('Sepal length')
    #     ax.set_ylabel('Sepal width')
    #     ax.set_xticks(())
    #     ax.set_yticks(())
    #     ax.set_title(title)
    #
    # plt.show()

