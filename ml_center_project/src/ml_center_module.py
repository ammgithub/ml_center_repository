"""
Created on October 18, 2017

Machine learning and another center (other than the analytic center)

"""

import numpy as np
from scipy.optimize import linprog as lp
from sklearn import svm
from sklearn.metrics.pairwise import pairwise_kernels
import matplotlib.pyplot as plt
import warnings

__author__ = 'amm'
__date__ = "Oct 18, 2017"
__version__ = 0.0

np.set_printoptions(linewidth=100, edgeitems='all', suppress=True,
                    precision=4)

class FastKernelClassifier(object):
    """
    A very fast kernel machine.

    FastKernelClassifier requires Gurobi in order to run the method fit_grb().
    Otherwise use fit(), with significantly reduced performance.
    This includes not finding the optimal vector of weights in some instances.

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
    >>> fkc = FastKernelClassifier()
    >>> fkc = FastKernelClassifier(kernel='poly', degree=3, gamma=1, coef0=1)

    Elaborate example (OR problem)
    ------------------------------
    >>> trX = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
    >>> trY = [1, -1, 1, -1]
    >>> tsX = np.array([[1, 2], [-3, 2], [6, -1]])
    >>> tsY = [1, -1, 1]
    >>> kernel = 'rbf'; degree = 2; gamma = 1; coef0 = 1
    >>> fkc = FastKernelClassifier(kernel=kernel, degree=degree, gamma=gamma, coef0=coef0)
    >>> fkc.fit(trX, trY)
    >>> ftest = fkc.predict(tsX)
    >>> fkc.plot2d(0.02)

    Notes
    -----

    References
    ----------
    """
    def __init__(self, kernel='linear', degree=3, gamma=1, coef0=0.0, Csoft=10000.0,
                 verbose=False):
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.Csoft = Csoft

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
        c = np.vstack((np.zeros((self.num_train_samples+1, 1)),
                       self.Csoft*np.ones((self.num_train_samples, 1)), 1)).flatten()        # soft 5/7

        # Constraints from data (halfspaces)
        kmat = get_label_adjusted_train_kernel(trainx, trainy,
                                               kernel=self.kernel,
                                               degree=self.degree,
                                               gamma=self.gamma,
                                               coef0=self.coef0)

        # self.kmat = kmat
        # print "self.trainx = \n", self.trainx
        # print "kmat = \n", kmat
        Aub_data = np.hstack((-kmat, -np.eye(self.num_train_samples),           # soft 1/7
                              -np.ones((self.num_train_samples, 1))))
        bub_data = np.zeros((self.num_train_samples, 1))
        # print "Aub_data = \n", Aub_data

        # Box constraints lower
        Aub_box_lower = np.hstack((-np.identity(self.num_train_samples+1),
                                   np.zeros((self.num_train_samples + 1, self.num_train_samples)),  # soft 2/7
                                   -np.ones((self.num_train_samples+1, 1))))
        bub_box_lower = np.ones((self.num_train_samples+1, 1))

        # Box constraints upper
        Aub_box_upper = np.hstack((np.identity(self.num_train_samples+1),
                                   np.zeros((self.num_train_samples + 1, self.num_train_samples)),  # soft 3/7
                                   -np.ones((self.num_train_samples+1, 1))))
        bub_box_upper = np.ones((self.num_train_samples+1, 1))

        # Box xi                                                                soft 4/7
        Aub_box_xi = np.hstack((np.zeros((self.num_train_samples, self.num_train_samples + 1)),
                                -np.identity(self.num_train_samples),
                                np.zeros((self.num_train_samples, 1))))
        bub_box_xi = np.zeros((self.num_train_samples, 1))

        # Putting it all together
        Aub = np.vstack((Aub_data, Aub_box_lower, Aub_box_upper, Aub_box_xi))       # soft 6/7
        bub = np.vstack((bub_data, bub_box_lower, bub_box_upper, bub_box_xi)).flatten()

        # # Using scipy lp solver
        res = lp(c=c, A_ub=Aub, b_ub=bub, bounds=(None, None))
        weight_opt = res.x
        print "weight_opt = \n", weight_opt
        # last element is epsilon
        # self.weight_opt = weight_opt[:-1]
        self.weight_opt = weight_opt[:-(1+self.num_train_samples)]                       # soft 7/7
        self.pen_opt = weight_opt[(1+self.num_train_samples):-1]
        self.eps_opt = weight_opt[-1]
        self.fun_opt = res.fun
        if np.abs(self.eps_opt) <= 0.00001:
            warnings.warn('\neps_opt is close to zero. Data not separable. ')

    def fit_grb(self, trainx, trainy):
        """
        Same as the fit method, however, using Gurobi.
        Compute the optimal weight vector to classify (trainx, trainy) using Gurobi.

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
        import gurobipy as grb
        [self.num_train_samples, self.num_features] = trainx.shape

        # Need trainx and testx in 'predict', need trainy in 'plot2d'
        self.trainx = trainx
        self.trainy = trainy

        # Objective function
        c = np.vstack((np.zeros((self.num_train_samples+1, 1)),
                       self.Csoft*np.ones((self.num_train_samples, 1)), 1)).flatten()        # soft 5/7

        # Constraints from data (halfspaces)
        kmat = get_label_adjusted_train_kernel(trainx, trainy,
                                               kernel=self.kernel,
                                               degree=self.degree,
                                               gamma=self.gamma,
                                               coef0=self.coef0)

        # self.kmat = kmat
        # print "self.trainx = \n", self.trainx
        # print "kmat = \n", kmat
        Aub_data = np.hstack((-kmat, -np.eye(self.num_train_samples),           # soft 1/7
                              -np.ones((self.num_train_samples, 1))))
        bub_data = np.zeros((self.num_train_samples, 1))
        # print "Aub_data = \n", Aub_data

        # Box constraints lower
        Aub_box_lower = np.hstack((-np.identity(self.num_train_samples+1),
                                   np.zeros((self.num_train_samples + 1, self.num_train_samples)),  # soft 2/7
                                   -np.ones((self.num_train_samples+1, 1))))
        bub_box_lower = np.ones((self.num_train_samples+1, 1))

        # Box constraints upper
        Aub_box_upper = np.hstack((np.identity(self.num_train_samples+1),
                                   np.zeros((self.num_train_samples + 1, self.num_train_samples)),  # soft 3/7
                                   -np.ones((self.num_train_samples+1, 1))))
        bub_box_upper = np.ones((self.num_train_samples+1, 1))

        # Box xi                                                                soft 4/7
        Aub_box_xi = np.hstack((np.zeros((self.num_train_samples, self.num_train_samples + 1)),
                                -np.identity(self.num_train_samples),
                                np.zeros((self.num_train_samples, 1))))
        bub_box_xi = np.zeros((self.num_train_samples, 1))

        # Putting it all together
        Aub = np.vstack((Aub_data, Aub_box_lower, Aub_box_upper, Aub_box_xi))       # soft 6/7
        bub = np.vstack((bub_data, bub_box_lower, bub_box_upper, bub_box_xi)).flatten()

        # Using Gurobi
        m = grb.Model()

        # Switch off console output
        # m.Params.OutputFlag = 1
        m.setParam('OutputFlag', 1)
        m.setParam('Method', 1)
        m.setParam('TimeLimit', 1)
        m.setParam('Aggregate', 0)

        # m.set(grb.GRB_IntParam_OutputFlag, 0)
        J = range(2 * self.num_train_samples + 2)
        I = range(4 * self.num_train_samples + 2)
        x = [m.addVar(lb=-grb.GRB.INFINITY, ub=grb.GRB.INFINITY,
                      obj=c[j], vtype=grb.GRB.CONTINUOUS,
                      name="weight " + str(j + 1)) for j in J]  # TODO: add name, continuous explicit?
        m.update()
        constraints = [m.addConstr(grb.quicksum(Aub[i, j] * x[j] for j in J) <= bub[i]) for i in I]
        m.update()

        m.optimize()
        weight_opt = [x[j].x for j in J]
        # last element is epsilon
        # self.weight_opt = weight_opt[:-1]
        self.weight_opt = weight_opt[:-(1+self.num_train_samples)]                       # soft 7/7
        self.pen_opt = weight_opt[(1+self.num_train_samples):-1]
        self.eps_opt = weight_opt[-1]
        self.fun_opt = m.objval
        if np.abs(self.eps_opt) <= 0.00001:
            warnings.warn('\neps_opt is close to zero. Data not separable. ')
        if m.Status != 2:
            warnings.warn('\nGurobi did not return an optimal solution. ')

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
        # self.ktest = ktest
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
            # colormap is coolwarm
            out = ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
            ax.scatter(self.trainx[:, 0], self.trainx[:, 1], c=self.trainy,
                       cmap=plt.cm.coolwarm, s=60, edgecolors='k')
            ax.set_xlabel('trainx[:, 0] - Attribute 1')
            ax.set_ylabel('trainx[:, 1] - Attribute 2')
            title_string = "FKC - Training data and decision surface for: \nKernel = %s, " \
                           "degree =  %1.1f, gamma =  %1.1f, coef0 =  %1.1f, Csoft =  %4.4f" % (
                            self.kernel, self.degree, self.gamma, self.coef0, self.Csoft)
            ax.set_title(title_string)
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

    # Testing extended CIRCLE
    print "FKC: Testing extended circular problem \n"
    trX = np.array([[1, 1], [4, 1], [1, 4], [4, 4], [2, 2], [2, 3], [3, 2], [5, 4.5]])
    trY = [1, 1, 1, 1, -1, -1, -1, -1]
    tsX = np.array([[0, 2], [3, 3], [6, 3]])
    tsY = [1, -1, 1]
    kernel = 'poly'; degree = 2; gamma = 1; coef0 = 1; Csoft = 0.1

    print "kernel = %s, degree = %d, gamma = %3.2f, coef0 = %3.2f, Csoft = %5.4f" \
          % (kernel, degree, gamma, coef0, Csoft)
    print "-----------------------------------------------------"
    fkc = FastKernelClassifier(kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, Csoft=Csoft)
    fkc.fit_grb(trX, trY)
    # fkc.fit(trX, trY)
    print "fkc.eps_opt = ", fkc.eps_opt
    print "fkc.weight_opt  (l+1-vector) = \n", fkc.weight_opt
    print "fkc.pen_opt (l-vector) = \n", fkc.pen_opt
    print "fkc.fun_opt = ", fkc.fun_opt
    ftest = fkc.predict(tsX)
    print "tsX = \n", tsX
    print "fkc.predict(tsX) = \n", ftest
    print "tsY = \n", tsY
    if not (abs(ftest - tsY) <= 0.001).all():
        print "*** Test set not classified correctly. ***"
    ftest = fkc.predict(trX)
    print "trX = \n", trX
    print "fkc.predict(trX) = \n", ftest
    print "trY = \n", trY
    if not (abs(ftest - trY) <= 0.001).all():
        print "*** TRAINING SET NOT CLASSIFIED CORRECTLY. ***"
    fkc.plot2d(0.02)

    # # Include a small dataset to run module: OR problem
    # print "Testing OR:"
    # trX = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
    # trY = [1, -1, 1, -1]
    # tsX = np.array([[1, 2], [-3, 2], [6, -1]])
    # tsY = [1, -1, 1]
    # kernel = 'poly'; degree = 2; gamma = 1; coef0 = 1; Csoft = 10
    #
    # print "kernel = %s, degree = %d, gamma = %3.2f, coef0 = %3.2f, Csoft = %5.1f" \
    #       % (kernel, degree, gamma, coef0, Csoft)
    # print "-----------------------------------------------------"
    # fkc = FastKernelClassifier(kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, Csoft=Csoft)
    # fkc.fit_grb(trX, trY)
    # # fkc.fit(trX, trY)
    # print "fkc.eps_opt = ", fkc.eps_opt
    # print "fkc.weight_opt  (l+1-vector) = \n", fkc.weight_opt
    # print "fkc.pen_opt (l-vector) = \n", fkc.pen_opt
    # print "fkc.fun_opt = ", fkc.fun_opt
    # ftest = fkc.predict(tsX)
    # print "tsX = \n", tsX
    # print "fkc.predict(tsX) = \n", ftest
    # print "tsY = \n", tsY
    # if not (abs(ftest - tsY) <= 0.001).all():
    #     print "*** Test set not classified correctly. ***"
    # ftest = fkc.predict(trX)
    # print "trX = \n", trX
    # print "fkc.predict(trX) = \n", ftest
    # print "trY = \n", trY
    # if not (abs(ftest - trY) <= 0.001).all():
    #     print "*** TRAINING SET NOT CLASSIFIED CORRECTLY. ***"
    # fkc.plot2d(0.02)
