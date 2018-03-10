"""
Created on November 29, 2017

Machine learning and another center (other than the analytic center)

"""

import numpy as np
from scipy.optimize import linprog as lp
from sklearn.metrics.pairwise import pairwise_kernels
import matplotlib.pyplot as plt
import warnings
import gurobipy as grb

__author__ = 'amm'
__date__ = "Oct 18, 2017"
__version__ = 0.0

np.set_printoptions(linewidth=100, edgeitems=None, suppress=True,
                    precision=4)


class FastKernelClassifier(object):
    """
    A very fast kernel machine.

    FastKernelClassifier requires Gurobi in order to run the method fit_grb().
    Otherwise use fit(), with significantly reduced performance.
    This includes not finding the optimal vector of weights in some instances.

    The training and testing data are included in the object as "None', they are
    defined more precisely in the methods:

    fit, fit_grb, fit_hard, fit_grb_hard, predict, and score.

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

    Csoft : float, (default=10000.0)
    Penalty parameter for soft margin

    Examples
    --------
    >>> fkc = FastKernelClassifier(kernel='poly', degree=3, gamma=1, coef0=1, Csoft=10000)

    Elaborate example (OR problem)
    ------------------------------
    >>> trX = np.array([[1, 1], [-1, 1], [1, -1], [-1, -1]])
    >>> trY = [1, -1, -1, 1]
    >>> tsX = np.array([[1, 2], [-3, 2], [6, -1]])
    >>> tsY = [1, -1, 1]
    >>> kernel = 'rbf'; degree = 2; gamma = 1; coef0 = 1; Csoft = 10000
    >>> fkc = FastKernelClassifier(kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, Csoft=Csoft)
    >>> fkc.fit(trX, trY)
    >>> ftest = fkc.predict(tsX)
    >>> fkc.plot2d()

    Notes
    -----

    References
    ----------
    """
    def __init__(self, kernel='linear', degree=3, gamma=1.0, coef0=0.0, Csoft=10000.0):
        """
        :param kernel:  Type of kernel, 'poly', 'rbf', default is 'linear'
        :type kernel:   string

        :param degree:  Degree of the polynomial kernel. Ignored by other kernels
        :type degree:   int

        :param gamma:   Radial-basis function kernels. gamma = 1 / 2 sigma^2.
                        polynomial kernels: gamma is a multiplier of u*v
        :type gamma:    float

        :param coef0:   Bias for polynomial kernels only, default = 0.0
        :type coef0:    float

        :param Csoft:   Penalty parameter for soft margin, default = 10000.0
        :type Csoft:    float
        """
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.Csoft = Csoft

        self.trainx = None
        self.trainy = None
        self.num_train_samples = None
        self.num_features = None

        self.weight_opt = None
        self.pen_opt = None
        self.eps_opt = None
        self.fun_opt = None

        # plot2d: run problems without training data
        self.testx = np.zeros((2, 2))
        self.testy = np.zeros(2)

    def fit(self, trainx, trainy):
        """
        Soft margin case:
        Compute the optimal weight vector to classify (trainx, trainy) using the
        Scipy function 'linprog'.

        The variable assignment for l = 4 samples is given by

        x = (alpha_1, alpha_2, alpha_3, alpha_4, b, xi_1, xi_2, xi_3, xi_4, eps)

        leading to 2*l + 2 = 10 variables.

        Parameters
        ----------
        :param trainx:      Input training samples, num_train_samples-by-num_features
        :type trainx:       numpy array of floats

        :param trainy:      Input training labels, num_train_samples-by-one
        :type trainy:       numpy array of ints

        Returns
        -------
        self : object

        """
        print "Fitting soft margin using Scipy linprog...\n"
        [self.num_train_samples, self.num_features] = trainx.shape

        # trainx and testx in 'predict', trainy for kernel, and testy for 'plot2d'
        self.trainx = trainx
        self.trainy = trainy

        # Constraints from data (halfspaces), this is the transpose of K
        Ktraintrans = get_label_adjusted_train_kernel(trainx, trainy,
                                                      kernel=self.kernel,
                                                      degree=self.degree,
                                                      gamma=self.gamma,
                                                      coef0=self.coef0)

        # Objective function
        c = np.vstack((np.zeros((self.num_train_samples+1, 1)),
                       self.Csoft*np.ones((self.num_train_samples, 1)), 1)).flatten()        # soft 5/7

        a_ub = np.hstack((-Ktraintrans, -np.eye(self.num_train_samples),           # soft 1/7
                          -np.ones((self.num_train_samples, 1))))
        # b_ub = np.zeros((self.num_train_samples, 1))
        b_ub = np.zeros(self.num_train_samples)
        lb = np.hstack((-np.ones(self.num_train_samples + 1),
                        np.zeros(self.num_train_samples), -1e9))
        ub = np.hstack((np.ones(self.num_train_samples + 1),
                        1e9*np.ones(self.num_train_samples + 1)))

        # Scipy lp solver: use bland option?)
        result = lp(c=c, A_ub=a_ub, b_ub=b_ub, bounds=zip(lb, ub),
                 options=dict(bland=True))

        if result.message == 'Optimization failed. Unable to ' \
                             'find a feasible starting point.':
            print result
        if result.status > 0:
            print 'Scipy linprog did not terminate successfully. status = %d' \
                  % result.status

        weight_opt = result.x
        self.weight_opt = weight_opt[:-(1+self.num_train_samples)]                       # soft 7/7
        self.pen_opt = weight_opt[(1+self.num_train_samples):-1]
        self.eps_opt = weight_opt[-1]
        self.fun_opt = result.fun
        if np.abs(self.eps_opt - self.fun_opt) >= 0.00001:
            warnings.warn('\neps_opt is not identical to fun_opt. eps_opt - fun_opt = %0.6f. ' \
                          % (self.eps_opt - self.fun_opt))
        if np.abs(self.eps_opt) <= 0.00001:
            warnings.warn('\neps_opt is close to zero. Data not separable. ')

    def fit_hard(self, trainx, trainy):
        """
        Hard margin case (mainly for testing):
        Compute the optimal weight vector to classify (trainx, trainy) using the
        Scipy function 'linprog'.

        The variable assignment for l = 4 samples is given by

        x = (alpha_1, alpha_2, alpha_3, alpha_4, b, eps)

        leading to l + 2 = 6 variables.

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
        print "Fitting hard margin using Scipy linprog...\n"
        [self.num_train_samples, self.num_features] = trainx.shape

        # trainx and testx in 'predict', trainy for kernel, and testy for 'plot2d'
        self.trainx = trainx
        self.trainy = trainy

        # Constraints from data (halfspaces), this is the transpose of K
        Ktraintrans = get_label_adjusted_train_kernel(trainx, trainy,
                                                      kernel=self.kernel,
                                                      degree=self.degree,
                                                      gamma=self.gamma,
                                                      coef0=self.coef0)

        # Objective function
        c = np.vstack((np.zeros((self.num_train_samples+1, 1)), 1)).flatten()

        a_ub = np.hstack((-Ktraintrans, -np.ones((self.num_train_samples, 1))))
        # b_ub = np.zeros((self.num_train_samples, 1))
        b_ub = np.zeros(self.num_train_samples)
        lb = np.hstack((-np.ones(self.num_train_samples + 1), -1e9))
        ub = np.hstack((np.ones(self.num_train_samples + 1), 1e9))

        # Scipy lp solver: use bland option?)
        result = lp(c=c, A_ub=a_ub, b_ub=b_ub, bounds=zip(lb, ub),
                 options=dict(bland=True))

        if result.message == 'Optimization failed. Unable to ' \
                             'find a feasible starting point.':
            print result
        if result.status > 0:
            print 'Scipy linprog did not terminate successfully. status = %d' \
                  % result.status

        weight_opt = result.x
        self.weight_opt = weight_opt[:-1]
        self.eps_opt = weight_opt[-1]
        self.fun_opt = result.fun
        if np.abs(self.eps_opt - self.fun_opt) >= 0.00001:
            warnings.warn('\neps_opt is not identical to fun_opt. eps_opt - fun_opt = %0.6f. ' \
                          % (self.eps_opt - self.fun_opt))
        if np.abs(self.eps_opt) <= 0.00001:
            warnings.warn('\neps_opt is close to zero. Data not separable. ')

    def fit_grb(self, trainx, trainy):
        """
        Soft margin case:
        Same as the 'fit' method, however, using Gurobi.
        Compute the optimal weight vector to classify (trainx, trainy) using Gurobi.

        The variable assignment for l = 4 samples is given by

        x = (alpha_1, alpha_2, alpha_3, alpha_4, b, xi_1, xi_2, xi_3, xi_4, eps)

        leading to 2*l + 2 = 10 variables.

        Parameters
        ----------
        trainx : numpy array of floats, num_train_samples-by-num_features
                 Input training samples

        trainy : list of numpy array of floats or integers num_train_samples-by-one
                 Training labels

        Returns
        -------
        self : object

        """
        print "Fitting soft margin using Gurobi...\n"
        [self.num_train_samples, self.num_features] = trainx.shape

        # trainx and testx in 'predict', trainy for kernel, and testy for 'plot2d'
        self.trainx = trainx
        self.trainy = trainy

        # Constraints from data (halfspaces), this is the transpose of K
        Ktraintrans = get_label_adjusted_train_kernel(trainx, trainy,
                                                      kernel=self.kernel,
                                                      degree=self.degree,
                                                      gamma=self.gamma,
                                                      coef0=self.coef0)

        # Objective function
        c = np.vstack((np.zeros((self.num_train_samples+1, 1)),
                       self.Csoft*np.ones((self.num_train_samples, 1)), 1)).flatten()        # soft 5/7

        a_ub = np.hstack((-Ktraintrans, -np.eye(self.num_train_samples),           # soft 1/7
                          -np.ones((self.num_train_samples, 1))))
        b_ub = np.zeros(self.num_train_samples)
        lb = np.hstack((-np.ones(self.num_train_samples + 1),
                        np.zeros(self.num_train_samples), -1e9))
        ub = np.hstack((np.ones(self.num_train_samples + 1),
                        1e9*np.ones(self.num_train_samples + 1)))

        # Speed improvement in Gurobi due to as dictionary inconclusive
        a_ub_dict = {i: {j: v for j, v in enumerate(row)}
                     for i, row in enumerate(a_ub)}

        # Using Gurobi
        m = grb.Model()

        # Switch off console output
        m.setParam('OutputFlag', 0)
        m.setParam('Method', 1)  # dual simplex
        # m.setParam('TimeLimit', 1)
        # m.setParam('Aggregate', 0)

        # m.set(grb.GRB_IntParam_OutputFlag, 0)
        var_list = range(2 * self.num_train_samples + 2)

        # Using a list
        # xold = [m.addVar(lb=lb[j], ub=ub[j], obj=c[j],
        #                  vtype=grb.GRB.CONTINUOUS) for j in var_list]

        # Using a dictionary, but speed-up inconclusive
        x = {j: m.addVar(lb=lb[j], ub=ub[j], obj=c[j], vtype=grb.GRB.CONTINUOUS)
             for j in var_list}
        m.update()

        constr_list = range(self.num_train_samples)
        for i in constr_list:
            m.addConstr(grb.quicksum(x[j] * a_ub_dict[i][j]
                                     for j in var_list) <= b_ub[i])
            # m.addConstr(grb.quicksum(x[j] * a_ub[i, j]
            #                          for j in var_list) <= b_ub[i])
        m.optimize()

        # pr = cProfile.Profile()
        # pr.enable()
        # print "+++++++++++++++++++++++++++++++++++++++++++"
        # pr.disable()
        # pr.print_stats(sort='time')

        if m.Status != 2:
            warnings.warn('\nGurobi did not return an optimal solution. ')

        self.m = m

        weight_opt = [x[j].x for j in var_list]
        self.weight_opt = weight_opt[:-(1+self.num_train_samples)]                       # soft 7/7
        self.pen_opt = weight_opt[(1+self.num_train_samples):-1]
        self.eps_opt = weight_opt[-1]
        self.fun_opt = m.objval
        if np.abs(self.eps_opt - self.fun_opt) >= 0.00001:
            warnings.warn('\neps_opt is not identical to fun_opt. eps_opt - fun_opt = %0.6f. ' \
                          % (self.eps_opt - self.fun_opt))
        if np.abs(self.eps_opt) <= 0.00001:
            warnings.warn('\neps_opt is close to zero. Data not separable. ')

    def fit_grb_hard(self, trainx, trainy):
        """
        Hard margin case:
        Same as the 'fit_hard' method, however, using Gurobi.
        Compute the optimal weight vector to classify (trainx, trainy) using Gurobi.

        The variable assignment for l = 4 samples is given by

        x = (alpha_1, alpha_2, alpha_3, alpha_4, b, eps)

        leading to l + 2 = 6 variables.

        Parameters
        ----------
        trainx : numpy array of floats, num_train_samples-by-num_features
                 Input training samples

        trainy : list of numpy array of floats or integers num_train_samples-by-one
                 Training labels

        Returns
        -------
        self : object

        """
        print "Fitting hard margin using Gurobi...\n"
        [self.num_train_samples, self.num_features] = trainx.shape

        # trainx and testx in 'predict', trainy for kernel, and testy for 'plot2d'
        self.trainx = trainx
        self.trainy = trainy

        # Constraints from data (halfspaces), this is the transpose of K
        Ktraintrans = get_label_adjusted_train_kernel(trainx, trainy,
                                                      kernel=self.kernel,
                                                      degree=self.degree,
                                                      gamma=self.gamma,
                                                      coef0=self.coef0)

        # Objective function
        c = np.vstack((np.zeros((self.num_train_samples+1, 1)), 1)).flatten()

        a_ub = np.hstack((-Ktraintrans, -np.ones((self.num_train_samples, 1))))
        b_ub = np.zeros(self.num_train_samples)
        lb = np.hstack((-np.ones(self.num_train_samples + 1), -1e9))
        ub = np.hstack((np.ones(self.num_train_samples + 1), 1e9))

        # Speed improvement in Gurobi due to as dictionary inconclusive
        a_ub_dict = {i: {j: v for j, v in enumerate(row)}
                     for i, row in enumerate(a_ub)}

        # Using Gurobi
        m = grb.Model()

        # Switch off console output
        m.setParam('OutputFlag', 0)
        m.setParam('Method', 1)  # dual simplex
        # m.setParam('TimeLimit', 1)
        # m.setParam('Aggregate', 0)

        # m.set(grb.GRB_IntParam_OutputFlag, 0)
        var_list = range(self.num_train_samples + 2)

        # Using a list
        # xold = [m.addVar(lb=lb[j], ub=ub[j], obj=c[j],
        #                  vtype=grb.GRB.CONTINUOUS) for j in var_list]

        # Using a dictionary, but speed-up inconclusive
        x = {j: m.addVar(lb=lb[j], ub=ub[j], obj=c[j], vtype=grb.GRB.CONTINUOUS)
             for j in var_list}
        m.update()

        constr_list = range(self.num_train_samples)
        for i in constr_list:
            m.addConstr(grb.quicksum(x[j] * a_ub_dict[i][j]
                                     for j in var_list) <= b_ub[i])
            # m.addConstr(grb.quicksum(x[j] * a_ub[i, j]
            #                          for j in var_list) <= b_ub[i])
        m.optimize()

        if m.Status != 2:
            warnings.warn('\nGurobi did not return an optimal solution. ')

        self.m = m

        weight_opt = [x[j].x for j in var_list]
        self.weight_opt = weight_opt[:-1]
        self.eps_opt = weight_opt[-1]
        self.fun_opt = m.objval
        if np.abs(self.eps_opt - self.fun_opt) >= 0.00001:
            warnings.warn('\neps_opt is not identical to fun_opt. eps_opt - fun_opt = %0.6f. ' \
                          % (self.eps_opt - self.fun_opt))
        if np.abs(self.eps_opt) <= 0.00001:
            warnings.warn('\neps_opt is close to zero. Data not separable. ')

    def predict(self, testx):
        """
        Predict functional values of testx using weight_opt computed in 'fit'
        and 'fit_grb'.


        Parameters
        ----------
        testx :  numpy array of floats, l-by-num_features
                 Input test samples

        Returns
        -------
        self : object

        Notes
        -----
        Since 'predict' computes labels for some 'testx', which can be a single
        point, a meshgrid, or anything in-between, we can not define the formal
        test set 'self.testx' here.

        """
        self.num_test_samples = testx.shape[0]
        if np.abs(self.eps_opt) <= 0.00001:
            warnings.warn('\neps_opt is close to zero. Data not separable. ')
        Ktesttrans = get_label_adjusted_test_kernel(self.trainx, testx,
                                                    kernel=self.kernel,
                                                    degree=self.degree,
                                                    gamma=self.gamma,
                                                    coef0=self.coef0)

        return np.sign(np.dot(Ktesttrans, self.weight_opt))

    def score(self, testx, testy):
        """
        Computes the accuracy for a clearly defined test set
        consisting of inputs 'testx' and labels 'testy'.

        :param testx:   inputs for test set (n_test by l)
        :type testx:    array-like

        :param testy:   labels for test set (n_test by n_features)
        :type testy:    array-like

        """
        self.testx = testx
        self.testy = testy

        # accuracy: multiplication by 1 converts True/False to 1/0
        return sum((testy == self.predict(testx)) * 1) / float(len(testy))

    def score_train(self):
        """
        FKC: Verifies the accuracy of the training date separation.

        """
        if not (abs(self.predict(self.trainx) - self.trainy) <= 0.001).all():
            print "*** TRAINING SET NOT CLASSIFIED CORRECTLY. ***"

        # accuracy: multiplication by 1 converts True/False to 1/0
        return sum((self.trainy == self.predict(self.trainx)) * 1) /\
            float(len(self.trainy))

    def plot2d(self, this_title_info=' ', meshstep=0.02):
        """
        Plot simple examples that have 2-dimensional input training samples
        (2 features)

        Parameters
        ----------
        meshstep    :   float, (default=0.02)
                        Precision in meshgrid, smaller values result in smoother functions.

        this_title_info :    Additional information that can be passed on to figure
                        string

        Returns
        -------
        self : object
        """
        if self.trainx.shape[1] == 2:
            x1min = min(self.trainx[:, 0].min() - 3, self.testx[:, 0].min() - 3)
            x1max = max(self.trainx[:, 0].max() + 3, self.testx[:, 0].max() + 3)
            x2min = min(self.trainx[:, 1].min() - 3, self.testx[:, 1].min() - 3)
            x2max = max(self.trainx[:, 1].max() + 3, self.testx[:, 1].max() + 3)
            xx1, xx2 = np.meshgrid(np.arange(x1min, x1max + meshstep, meshstep),
                                 np.arange(x2min, x2max + meshstep, meshstep))

            fig, ax = plt.subplots(1, 1)
            xx = np.c_[xx1.ravel(), xx2.ravel()]
            # This line overwrites self.testx
            z = self.predict(xx)

            z = z.reshape(xx1.shape)
            # colormap is coolwarm
            out = ax.contourf(xx1, xx2, z, cmap=plt.cm.coolwarm, alpha=0.8)
            ax.scatter(self.trainx[:, 0], self.trainx[:, 1], c=self.trainy,
                       cmap=plt.cm.coolwarm, s=80, marker='x', edgecolors='face')

            # Split the test set into correctly and incorrectly classified
            idx_good = (self.testy == self.predict(self.testx)) * 1
            idx_bad = ~(self.testy == self.predict(self.testx)) * 1
            ax.scatter(self.testx[np.nonzero(idx_good), 0],
                       self.testx[np.nonzero(idx_good), 1],
                       c=np.array(self.testy)[np.nonzero(idx_good)[0]],
                       cmap=plt.cm.coolwarm, s=80, marker='.', edgecolors='face')
            ax.scatter(self.testx[np.nonzero(idx_bad), 0],
                       self.testx[np.nonzero(idx_bad), 1],
                       c=np.array(self.testy)[np.nonzero(idx_bad)[0]],
                       cmap=plt.cm.coolwarm, s=120, marker='$\odot$', edgecolors='face')

            ax.set_xlabel('trainx[:, 0] and testx[:, 0]    |    Attribute 1')
            ax.set_ylabel('trainx[:, 1] and testx[:, 1]    |    Attribute 2')
            if this_title_info[-9:] == 'soft fit:':
                title_string = this_title_info + " FKC - Training and test data and decision surface for: " \
                    "\nKernel = %s, degree =  %1.1f, gamma =  %1.1f, coef0 =  %1.1f, Csoft =  %4.4f" % (
                            self.kernel, self.degree, self.gamma, self.coef0, self.Csoft)
            else:
                # do not display Csoft
                title_string = this_title_info + " FKC - Training and test data and decision surface for: " \
                    "\nKernel = %s, degree =  %1.1f, gamma =  %1.1f, coef0 =  %1.1f" % (
                            self.kernel, self.degree, self.gamma, self.coef0)
            ax.set_title(title_string)
            plt.grid()
            plt.show()
        else:
            return "Input sample dimension must be equal to 2. Exiting. "


def get_label_adjusted_train_kernel(trainx, trainy, **params):
    """
    Compute the training kernel matrix (THE TRANSPOSE ACTUALLY).

    This matrix also takes labels into consideration.  The training kernel matrix has l+1 rows and l columns,
    where l is the number of samples in trainx and trainy.
    All columns are multiplied by the corresponding yj.  This implies that the l+1 row
    contains yj's only.  Corresponds to K.T from Trafalis, Malyscheff, ACM, 2002.

    Parameters
    ----------
    :param trainx: input samples for training set (d by l)
    :param trainy: labels for training set (d by 1) (flattened)

            params = {  kernel,
                        degree,
                        gamma,
                        coef0}
    :return: Ktrain.T

    Usage:
    ------
    Ktraintrans = get_label_adjusted_train_kernel(trainx, trainy, kernel='poly', degree=3, gamma=1, coef0=1)
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
                         'Exiting.' % params['kernel'])
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
    :param trainx: input samples for training set (d by l)
    :param testx: input samples for test set (d by num_test_samples)

            params = {  kernel,
                        degree,
                        gamma,
                        coef0}
    :return: Ktest.T

    Usage:
    ------
    Ktesttrans = get_label_adjusted_test_kernel(trainx, testx)
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
                         'Exiting.' % params['kernel'])

    # add row of ones
    Ktest = np.vstack((ktest, np.ones((1, num_test_samples))))
    return Ktest.T


def print_output(this_fkc, testx, testy, this_title_info):
    """
    FKC:  Simple output routine to display results and figure:

    """
    print "fkc.eps_opt = ", this_fkc.eps_opt
    print "fkc.weight_opt  (l+1-vector) = \n", this_fkc.weight_opt
    if this_title_info[-9:] == 'soft fit:':
        print "fkc.pen_opt (l-vector) = \n", this_fkc.pen_opt
    print "tr_accuracy = ", this_fkc.score_train()
    print "testx = \n", testx
    print "testy = \n", testy
    print "ftest = \n", this_fkc.predict(testx)
    print "ts_accuracy = ", this_fkc.score(testx, testy)
    this_fkc.plot2d(this_title_info=this_title_info)

if __name__ == '__main__':
    """
    execfile('ml_center_module.py')
    springer thank you: thankyou1710
    """
    import os
    os.chdir('C:\\Users\\amalysch\\PycharmProjects\\ml_center_repository\\ml_center_project\\src')

    print 80 * "-"
    print 18 * " " + "Quick tests: Machine learning center module"
    print 80 * "-"
    print 2 * " " + "(1) FKC: Testing OR problem"
    print 2 * " " + "(2) FKC: Testing extended 2-dimensional circular data"
    print 2 * " " + "(3) FKC: Testing another extended 2-dimensional circular data"
    print 80 * "-"

    user_in = 0
    bad_input = True
    while bad_input:
        try:
            user_in = int(raw_input("Make selection: "))
            bad_input = False
        except ValueError as e:
            print "%s is not a valid selection. Please try again. "\
                  % e.args[0].split(':')[1]

    print "\n\n"
    if user_in == 1:
        print "(1) FKC: Testing OR problem \n"
        # Swapping last two training samples affects 'fit':  eps_opt<>fun_opt
        trX = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
        trY = [1, -1, 1, -1]
        tsX = np.array([[1, 2], [-3, 2], [6, -1]])
        tsY = [1, -1, 1]

        kernel = 'poly'
        degree = 2
        gamma = 1
        coef0 = 1
        Csoft = 22222

        print "kernel = %s, degree = %d, gamma = %3.2f, coef0 = %3.2f, Csoft = %5.4f" \
              % (kernel, degree, gamma, coef0, Csoft)
        print "-" * 70
        fkc = FastKernelClassifier(kernel=kernel, degree=degree, gamma=gamma,
                                   coef0=coef0, Csoft=Csoft)

        title_info = 'Scipy linprog soft fit:'
        print "\n" + title_info
        print 25 * "-"
        fkc.fit(trX, trY)
        print_output(fkc, tsX, tsY, title_info)

        title_info = 'Gurobi soft fit:'
        print "\n" + title_info
        print 25 * "-"
        fkc.fit_grb(trX, trY)
        print_output(fkc, tsX, tsY, title_info)

        title_info = 'Scipy linprog hard fit:'
        print "\n" + title_info
        print 25 * "-"
        fkc.fit_hard(trX, trY)
        print_output(fkc, tsX, tsY, title_info)

        title_info = 'Gurobi hard fit:'
        print "\n" + title_info
        print 25 * "-"
        fkc.fit_grb_hard(trX, trY)
        print_output(fkc, tsX, tsY, title_info)

    elif user_in == 2:
        print "(2) FKC: Testing extended 2-dimensional circular data \n"
        # works with scipy (Scipy bug 'Optimization failed. Unable to find a feasible starting point.')
        # trX = np.array([[1, 1], [4, 1], [1, 4], [4, 4], [2, 2], [2, 3], [3, 2], [4, 5.5]])
        # trY = [1, 1, 1, 1, -1, -1, -1, -1]
        trX = np.array([[1, 1], [4, 1], [1, 4], [4, 4], [2, 2], [2, 3], [3, 2], [5, 4.]])
        trY = [1, 1, 1, 1, -1, -1, -1, -1]
        tsX = np.array([[0, 2], [3, 3], [6, 3]])
        tsY = [1, -1, 1]

        kernel = 'poly'
        degree = 2
        gamma = 1
        coef0 = 1
        Csoft = 10.0

        print "kernel = %s, degree = %d, gamma = %3.2f, coef0 = %3.2f, Csoft = %5.4f" \
              % (kernel, degree, gamma, coef0, Csoft)
        print "-" * 70
        fkc = FastKernelClassifier(kernel=kernel, degree=degree, gamma=gamma,
                                   coef0=coef0, Csoft=Csoft)

        title_info = 'Scipy linprog soft fit:'
        print "\n" + title_info
        print 25 * "-"
        fkc.fit(trX, trY)
        print_output(fkc, tsX, tsY, title_info)

        title_info = 'Gurobi soft fit:'
        print "\n" + title_info
        print 25 * "-"
        fkc.fit_grb(trX, trY)
        print_output(fkc, tsX, tsY, title_info)

        title_info = 'Scipy linprog hard fit:'
        print "\n" + title_info
        print 25 * "-"
        fkc.fit_hard(trX, trY)
        print_output(fkc, tsX, tsY, title_info)

        title_info = 'Gurobi hard fit:'
        print "\n" + title_info
        print 25 * "-"
        fkc.fit_grb_hard(trX, trY)
        print_output(fkc, tsX, tsY, title_info)

    elif user_in == 3:
        print "(3) FKC: Testing another extended 2-dimensional circular data \n"
        trX = np.array([[1, 1], [4, 1], [1, 4], [4, 4], [2, 2], [2, 3], [3, 2], [4, 5.5]])
        trY = [1, 1, 1, 1, -1, -1, -1, -1]
        tsX = np.array([[0, 2], [3, 3], [6, 3]])
        tsY = [1, -1, 1]

        kernel = 'poly'
        degree = 2
        gamma = 1
        coef0 = 1
        Csoft = 10.0

        print "kernel = %s, degree = %d, gamma = %3.2f, coef0 = %3.2f, Csoft = %5.4f" \
              % (kernel, degree, gamma, coef0, Csoft)
        print "-" * 70
        fkc = FastKernelClassifier(kernel=kernel, degree=degree, gamma=gamma,
                                   coef0=coef0, Csoft=Csoft)

        title_info = 'Scipy linprog soft fit:'
        print "\n" + title_info
        print 25 * "-"
        fkc.fit(trX, trY)
        print_output(fkc, tsX, tsY, title_info)

        title_info = 'Gurobi soft fit:'
        print "\n" + title_info
        print 25 * "-"
        fkc.fit_grb(trX, trY)
        print_output(fkc, tsX, tsY, title_info)

        title_info = 'Scipy linprog hard fit:'
        print "\n" + title_info
        print 25 * "-"
        fkc.fit_hard(trX, trY)
        print_output(fkc, tsX, tsY, title_info)

        title_info = 'Gurobi hard fit:'
        print "\n" + title_info
        print 25 * "-"
        fkc.fit_grb_hard(trX, trY)
        print_output(fkc, tsX, tsY, title_info)

    else:
        print "Invalid selection. Program terminating. "
