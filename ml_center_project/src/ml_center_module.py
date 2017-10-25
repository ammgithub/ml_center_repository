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
import warnings

np.set_printoptions(linewidth=100, edgeitems='all', suppress=True,
                    precision=4)


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
    A very fast kernel machine.

    The following kernels are employed:

    'linear'
    --------
    k(u, v) = u*v

    'poly'
    ------
    k(u, v) = ( \gamma*u*v + coef0 )^degree

    'rbf'
    -----
    k(u, v) = exp (-gamma*||u - v||^2)

    Parameters
    ----------
    kernel : string, 'linear', 'poly', 'rbf' (default='linear')
    Describes the kernel function.

    degree : int, (default=3)
    Degree of the polynomial kernel. Ignored by other kernels.

    gamma : int, (default=1)
    For radial-basis function kernels. gamma = 1 / 2 sigma^2.
    For polynomial kernels. gamma is a multiplier of u*v.

    coef0 : float, (default=0.0)
    For polynomial (and sigmoid) kernels only.
    For polynomial kernels coef0 is a bias added to u*v

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
        trainx : numpy array of floats, num_train_samples-by-num_features
                 Input training samples

        trainy : list of numpy array of floats or integers num_train_samples-by-one
                 Input training labels

        Returns
        -------
        self : object

        """
        [self.num_train_samples, self.num_features] = trainx.shape

        # Need trainx and testx in 'predict', need trainy in 'plot2d'
        self.trainx = trainx
        self.trainy = trainy

        # Objective function
        c = np.vstack((np.zeros((self.num_train_samples+1, 1)), 1)).flatten()

        # Constraints from data (halfspaces)
        # kmat = get_label_adjusted_train_kernel(trainx, trainy)
        kmat = get_label_adjusted_train_kernel(trainx, trainy,
                                                kernel=self.kernel,
                                                degree=self.degree,
                                                gamma=self.gamma,
                                                coef0=self.coef0)

        # print "self.trainx = \n", self.trainx
        # print "kmat = \n", kmat
        Aub_data = np.hstack((-kmat, -np.ones((self.num_train_samples, 1))))
        bub_data = np.zeros((self.num_train_samples, 1))

        # Box constraints lower
        Aub_box_lower = np.hstack((-np.identity(self.num_train_samples+1),
                                   -np.ones((self.num_train_samples+1, 1))))
        bub_box_lower = np.ones((self.num_train_samples+1, 1))

        # Box constraints upper
        Aub_box_upper = np.hstack((np.identity(self.num_train_samples+1),
                                   -np.ones((self.num_train_samples+1, 1))))
        bub_box_upper = np.ones((self.num_train_samples+1, 1))

        # Putting it all together
        Aub = np.vstack((Aub_data, Aub_box_lower, Aub_box_upper))
        bub = np.vstack((bub_data, bub_box_lower, bub_box_upper)).flatten()
        res = lp(c=c, A_ub=Aub, b_ub=bub, bounds=(None, None))
        weight_opt = res.x

        # last element is epsilon
        self.weight_opt = weight_opt[:-1]
        self.eps_opt = res.fun
        if np.abs(self.eps_opt) <= 0.00001:
            warnings.warn('\neps_opt is close to zero. Data not separable. ')

    def predict(self, testx):
        """
        Predict functional values of testx using weight_opt computed in 'fit'.

        Parameters
        ----------
        testx :  numpy array of floats, num_train_samples-by-num_features
                 Input test samples

        Returns
        -------
        self : object
        """
        if np.abs(self.eps_opt) <= 0.00001:
            warnings.warn('\neps_opt is close to zero. Data not separable. ')
        ktest = get_label_adjusted_test_kernel(self.trainx, testx,
                                                 kernel=self.kernel,
                                                 degree=self.degree,
                                                 gamma=self.gamma,
                                                 coef0=self.coef0)
        # print "self.trainx = \n", self.trainx
        # print "testx = \n", testx
        # print "ktest = \n", ktest
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
        if self.trainx.shape[1] == 2:
            xmin = self.trainx[:, 0].min() - 3
            xmax = self.trainx[:, 0].max() + 3
            ymin = self.trainx[:, 1].min() - 3
            ymax = self.trainx[:, 1].max() + 3
            xx, yy = np.meshgrid(np.arange(xmin, xmax + meshstep, meshstep),
                                 np.arange(ymin, ymax + meshstep, meshstep))

            fig, ax = plt.subplots(1, 1)
            testx = np.c_[xx.ravel(), yy.ravel()]
            Z = self.predict(testx)

            Z = Z.reshape(xx.shape)
            out = ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
            ax.scatter(self.trainx[:, 0], self.trainx[:, 1], c=self.trainy,
                       cmap=plt.cm.coolwarm, s=60, edgecolors='k')
            plt.grid()
            plt.show()
        else:
            return "Input sample dimension must be equal to 2. Exiting. "


def get_label_adjusted_train_kernel(trainx, trainy, **params):
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
            params = {  kernel,
                        degree,
                        gamma,
                        coef0}
    Out   : Ktrain.T

    Usage:
    ------
    Ktrain = get_label_adjusted_train_kernel(trainx, trainy, kernel='poly', degree=3, gamma=1, coef0=1)
    """
    # TODO: more elegant: what are the **kwds in pairwise_kernels()?
    if params['kernel'] == 'linear':
        ktrain = pairwise_kernels(X=trainx, metric=params['kernel'])
    elif params['kernel'] == 'poly':
        ktrain = pairwise_kernels(X=trainx, metric=params['kernel'],
                             gamma=params['gamma'], degree=params['degree'],
                             coef0=params['coef0'])
    elif params['kernel'] == 'rbf':
        ktrain = pairwise_kernels(X=trainx, metric=params['kernel'],
                             gamma=params['gamma'])
    else:
        raise ValueError('Please check the selected kernel (\'%s\'). '
                         'Exiting.' %params['kernel'])
    # multiply by labels and add row, same K as in Trafalis Malyscheff, ACM, 2002
    Ktrain = np.array([ktrain[i, :] * trainy for i in range(len(ktrain[0, :]))])
    Ktrain = np.vstack((Ktrain, trainy))
    return Ktrain.T

def get_label_adjusted_test_kernel(trainx, testx, **params):
    """
    Compute the test kernel matrix.  The test kernel matrix has l+1 rows and l columns,
    however, the l+1 row has 1s, not the labels yj. Columns are not multiplied by yj.

    Parameters
    ----------
    In    : trainx, input samples for training set (d by l)
            testx, input samples for test set (d by num_test_samples)
            params = {  kernel,
                        degree,
                        gamma,
                        coef0}
    Out   : Ktest.T

    Usage:
    ------
    Ktest = get_label_adjusted_test_kernel(trainx, testx)
    """
    num_test_samples = testx.shape[0]

    # TODO: more elegant: what are the **kwds in pairwise_kernels()?
    if params['kernel'] == 'linear':
        ktest = pairwise_kernels(X=trainx, Y=testx, metric=params['kernel'])
    elif params['kernel'] == 'poly':
        ktest = pairwise_kernels(X=trainx, Y=testx, metric=params['kernel'],
                                 gamma=params['gamma'], degree=params['degree'],
                                 coef0=params['coef0'])
    elif params['kernel'] == 'rbf':
        ktest = pairwise_kernels(X=trainx, Y=testx, metric=params['kernel'],
                                 gamma=params['gamma'])
    else:
        raise ValueError('Please check the selected kernel (\'%s\'). '
                         'Exiting.' %params['kernel'])

    # add row of ones
    Ktest = np.vstack((ktest, np.ones((1, num_test_samples))))
    return Ktest.T

if __name__ == '__main__':
    """
    execfile('ml_center_module.py')
    springer thank you: thankyou1710
    """
    import os
    os.chdir('C:\\Users\\amalysch\\PycharmProjects\\ml_center_repository\\ml_center_project\\src')

    # Testing OR data
    # print "Testing OR:"
    # trX = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
    # trY = [1, -1, 1, -1]
    # tsX = np.array([[1, 2], [-3, 2], [6, -1]])
    # tsY = [1, -1, 1]
    # Testing AND data
    # print "Testing AND:"
    # trX = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
    # trY = [1, -1, -1, -1]
    # tsX = np.array([[1, 2], [-3, 2], [6, -1]])
    # tsY = [1, -1, 1]
    # Testing CIRCLE data
    # print "\nTesting CIRCLE:"
    # trX = np.array([[1, 1], [4, 1], [1, 4], [4, 4], [2, 2], [2, 3], [3, 2]])
    # trY = [1, 1, 1, 1, -1, -1, -1]
    # tsX = np.array([[0, 2], [3, 3], [6, 3]])
    # tsY = [1, -1, 1]

    # Running OR, AND, and CIRCLE
    # kernel = 'rbf'
    # degree = 2
    # gamma = 1
    # coef0 = 1
    # print "kernel = %s, degree = %d, gamma = %3.2f, coef0 = %3.2f"%(kernel, degree, gamma, coef0)
    # print "-----------------------------------------------------"
    #
    # fkm = FastKernelMachine(kernel=kernel, degree=degree, gamma=gamma, coef0=coef0)
    # fkm.fit(trX, trY)
    # print "(fkm.weight_opt, fkm.eps_opt) = ", (fkm.weight_opt, fkm.eps_opt)
    # ftest = fkm.predict(tsX)
    # print "fkm.predict(tsX) = \n", ftest
    # print "tsY = \n", tsY
    # if not (abs(ftest - tsY) <= 0.001).all():
    #     print "*** Test set not classified correctly. ***"
    # ftest = fkm.predict(trX)
    # print "fkm.predict(trX) = \n", ftest
    # print "trY = \n", trY
    # if not (abs(ftest - trY) <= 0.001).all():
    #     print "*** TRAINING SET NOT CLASSIFIED CORRECTLY. ***"
    # fkm.plot2d(0.02)

    # Running IRIS
    # iris = datasets.load_iris()
    # # trX = iris.data[:100, [0, 1]]
    # trX = iris.data[:100, :]
    # trY = iris.target[:100]
    # trY = [i if i==1 else -1 for i in trY]
    #
    # kernel = 'linear'
    # degree = 1
    # gamma = 1
    # coef0 = 1
    # print "\nkernel = %s, degree = %d, gamma = %3.2f, coef0 = %3.2f"%(kernel, degree, gamma, coef0)
    # print "-----------------------------------------------------"
    #
    # fkm = FastKernelMachine(kernel=kernel, degree=degree, gamma=gamma, coef0=coef0)
    # fkm.fit(trX, trY)
    # print "(fkm.weight_opt, fkm.eps_opt) = ", (fkm.weight_opt, fkm.eps_opt)
    # ftest = fkm.predict(trX)
    # print "fkm.predict(trX) = ", ftest
    # print "trY = ", trY
    # if not (abs(ftest - trY) <= 0.001).all():
    #     print "*** Training set not classified correctly. ***"
    # fkm.plot2d(0.02)

    # Running BREAST CANCER
    bc_data = datasets.load_breast_cancer()
    trX = bc_data.data
    trY = bc_data.target
    trY = np.array([i if i == 1 else -1 for i in trY])

    kernel = 'rbf'
    degree = 4
    gamma = 1.0
    coef0 = 1
    print "\nkernel = %s, degree = %d, gamma = %3.2f, coef0 = %3.2f"%(kernel, degree, gamma, coef0)
    print "-----------------------------------------------------"

    fkm = FastKernelMachine(kernel=kernel, degree=degree, gamma=gamma, coef0=coef0)
    fkm.fit(trX, trY)
    print "(fkm.weight_opt, fkm.eps_opt) = ", (fkm.weight_opt, fkm.eps_opt)
    ftest = fkm.predict(trX)
    print "fkm.predict(trX) = \n", ftest
    print "trY = \n", trY
    if not (abs(ftest - trY) <= 0.001).all():
        print "*** TRAINING SET NOT CLASSIFIED CORRECTLY. ***"
    fkm.plot2d(0.02)
