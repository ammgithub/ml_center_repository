"""
Created on December 1, 2017

SVM classifier (sklearn built-in)

"""
import numpy as np
from sklearn import svm, datasets
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from time import time
import pickle

__author__ = 'amm'
__date__ = "Oct 31, 2017"
__version__ = 0.0

np.set_printoptions(precision=4, edgeitems=None, linewidth=100, suppress=True)


class MySVC(svm.SVC):
    """
    Parent class of sklearn's svm.SVC().

    Parameters
    ----------
    Mostly as svm.SVC(), but includes also the training and testing data
    in the object as "None', they are defined more precisely in the methods:

    fit, fit_grb, fit_hard, fit_grb_hard, predict, and score.

    Examples
    --------
    >> mysvc = MySVC(Csoft=1.0, cache_size=200, coef0=0.0, degree=3, gamma=1, kernel='rbf')

    Elaborate example (OR problem)
    ------------------------------
    >>> trX = np.array([[1, 1], [-1, 1], [1, -1], [-1, -1]])
    >>> trY = [1, -1, -1, 1]
    >>> tsX = np.array([[1, 2], [-3, 2], [6, -1]])
    >>> tsY = [1, -1, 1]
    >>> kernel = 'rbf'; degree = 2; gamma = 1; coef0 = 1; Csoft = 10000
    >>> mysvc = MySVC(kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, Csoft=Csoft)
    >>> mysvc.fit(trX, trY)
    >>> ftest = mysvc.predict(tsX)
    >>> mysvc.plot2d()
    """
    def __init__(self, Csoft=1.0, cache_size=200, coef0=0.0,
                 degree=3, gamma=1, kernel='rbf'):

        svm.SVC.__init__(self, C=Csoft, cache_size=cache_size, coef0=coef0,
                         degree=degree, gamma=gamma, kernel=kernel)

        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.cache_size = cache_size
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

    def load_data(self, trainx, trainy, testx, testy):
        """
        sklearn's built-in object svm.SVC object does not load the training data.
        In order to pass everything with the object include the data in the object.
        (Problems are not expected to be huge in size.)

        Parameters
        ----------
        :param trainx:      Input training samples, num_train_samples-by-num_features
        :type trainx:       numpy array of floats

        :param trainy:      Input training labels, num_train_samples-by-one
        :type trainy:       numpy array of ints

        :param testx:       Test samples, num_test_samples-by-num_features
        :type testx:        numpy array of floats

        :param testy:       Test labels, num_train_samples-by-one
        :type testy:        numpy array of ints
        """
        self.trainx = trainx
        self.trainy = trainy
        self.testx = testx
        self.testy = testy

    def score_train(self):
        """
        MySVC: Verifies the accuracy of the training date separation.

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
            # This line overwrites self.testx: # TODO: true for fkc, but also for svm.SVC()?
            z = self.predict(xx)

            z = z.reshape(xx1.shape)
            # colormap is coolwarm
            out = ax.contourf(xx1, xx2, z, cmap=plt.cm.coolwarm, alpha=0.8)
            ax.scatter(self.trainx[:, 0], self.trainx[:, 1], c=self.trainy,
                       cmap=plt.cm.coolwarm, s=80, marker='x', edgecolors='face')

            # Split the test set into correctly and incorrectly classified
            idx_good = (self.testy == self.predict(self.testx)) * 1
            idx_bad = ~(self.testy == self.predict(self.testx)) * 1
            ax.scatter(self.testx[np.nonzero(idx_good), 0][0],
                       self.testx[np.nonzero(idx_good), 1][0],
                       c=np.array(self.testy)[np.nonzero(idx_good)[0]],
                       cmap=plt.cm.coolwarm, s=80, marker='.', edgecolors='face')
            ax.scatter(self.testx[np.nonzero(idx_bad), 0][0],
                       self.testx[np.nonzero(idx_bad), 1][0],
                       c=np.array(self.testy)[np.nonzero(idx_bad)[0]],
                       cmap=plt.cm.coolwarm, s=120, marker='$\odot$', edgecolors='face')

            ax.set_xlabel('trainx[:, 0] and testx[:, 0]    |    Attribute 1')
            ax.set_ylabel('trainx[:, 1] and testx[:, 1]    |    Attribute 2')
            title_string = this_title_info + " SVC - Training and test data and decision surface for: " \
                "\nKernel = %s, degree =  %1.1f, gamma =  %1.1f, coef0 =  %1.1f, C =  %4.4f" % (
                        self.kernel, self.degree, self.gamma, self.coef0, self.Csoft)
            ax.set_title(title_string)
            plt.grid()
            plt.show()
        else:
            return "Input sample dimension must be equal to 2. Exiting. "


def print_output(this_mysvc, testx, testy, this_title_info):
    """
    MySVC:  Simple output routine to display results and figure:

    """
    # print "mysvc.weight_opt  (l+1-vector) = \n", this_mysvc.weight_opt
    # print "mysvc.pen_opt (l-vector) = \n", this_mysvc.pen_opt
    print "testx = \n", testx
    print "testy = \n", testy
    print "ftest = \n", this_mysvc.predict(testx)
    print "ts_accuracy = ", this_mysvc.score(testx, testy)
    this_mysvc.plot2d(this_title_info=this_title_info)

if __name__ == '__main__':
    """
    
    Examples
    --------
    >>> clf = svm.SVC(C=10000.0, cache_size=200, \
                  coef0=1, degree=2, gamma=12, kernel='rbf')
    >>> fkc = clf = svm.SVC()

    Elaborate example (OR problem)
    ------------------------------
    >>> trX = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
    >>> trY = [1, -1, 1, -1]
    >>> tsX = np.array([[1, 2], [-3, 2], [6, -1]])
    >>> tsY = [1, -1, 1]
    >>> kernel = 'rbf'; degree = 2; gamma = 1; coef0 = 1
    >>> svc_C = 10000.0;svc_kernel = 'rbf';svc_degree = 2;
    >>> svc_gamma = 12; svc_coef0 = 1;svc_cache_size = 200
    >>> clf = svm.SVC(C=svc_C, cache_size=svc_cache_size, \
                  coef0=svc_coef0, degree=svc_degree, \
                  gamma=svc_gamma, kernel=svc_kernel)
    >>> print clf
    >>> clf.fit(trX, trY)
    >>> ftest = clf.predict(tsX)
    
    execfile('svc_module.py')
    """
    import os
    os.chdir('C:\\Users\\amalysch\\PycharmProjects\\ml_center_repository\\ml_center_project\\src')

    print 80 * "-"
    print 18 * " " + "Support vector machine classification tests (Python built-in)"
    print 80 * "-"
    print 2 * " " + "(1) SVC: Testing OR problem"
    print 2 * " " + "(2) SVC: Testing AND problem"
    print 2 * " " + "(3) SVC: Testing 2-dimensional circular data"
    print 2 * " " + "(4) SVC: Testing extended 2-dimensional circular data"
    print 2 * " " + "(5) SVC: IRIS dataset: Testing 2-attribute, 2-class version (samples 0,...,99, classes 0, 1)"
    print 2 * " " + "(6) SVC: IRIS dataset: Testing 4-attribute, 2-class version (samples 0,...,99, classes 0, 1)"
    print 2 * " " + "(7) SVC: IRIS dataset: Computing generalization error for 2-class (100 experiments)"
    print 2 * " " + "(8) (x) SVC: BREAST CANCER dataset: Testing (all samples)"
    print 2 * " " + "(9) SVC: BREAST CANCER dataset: Computing generalization error (100 experiments)"
    print 2 * " " + "(11) SVC RBF: BREAST CANCER dataset: Computing accuracy for many (gamma, C) (takes a while)"
    print 2 * " " + "(12) SVC POLY: BREAST CANCER dataset: Computing accuracy for many (degree, C) (takes a while)"
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
        print "(1) SVC: Testing OR problem \n"
        # Testing OR
        trX = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
        trY = [1, -1, 1, -1]
        tsX = np.array([[1, 2], [-3, 2], [6, -1]])
        tsY = [1, -1, 1]

        svc_kernel = 'poly'
        svc_degree = 2
        svc_gamma = 1
        svc_coef0 = 1
        svc_cache_size = 200
        svc_C = 0.013

        # use: print mysvc
        print "kernel = %s, degree = %d, gamma = %3.2f, coef0 = %3.2f, svc_C = %5.4f"\
              % (svc_kernel, svc_degree, svc_gamma, svc_coef0, svc_C)
        print "-----------------------------------------------------"

        mysvc = MySVC(Csoft=svc_C, cache_size=svc_cache_size, coef0=svc_coef0,
                      degree=svc_degree, gamma=svc_gamma, kernel=svc_kernel)

        title_info = 'SVC soft fit:'
        print "\n" + title_info
        print 25 * "-"
        mysvc.fit(trX, trY)
        mysvc.load_data(trX, trY, tsX, tsY)
        print_output(mysvc, tsX, tsY, title_info)

    elif user_in == 2:
        print "(1) SVC: Testing AND problem \n"
        # Testing AND
        trX = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
        trY = [1, -1, -1, -1]
        tsX = np.array([[1, 2], [-3, 2], [6, -1]])
        tsY = [1, -1, 1]

        svc_kernel = 'poly'
        svc_degree = 5
        svc_gamma = 1
        svc_coef0 = 1
        svc_cache_size = 200
        svc_C = 10000.0

        # use: print mysvc
        print "kernel = %s, degree = %d, gamma = %3.2f, coef0 = %3.2f, svc_C = %5.4f"\
              % (svc_kernel, svc_degree, svc_gamma, svc_coef0, svc_C)
        print "-----------------------------------------------------"

        mysvc = MySVC(Csoft=svc_C, cache_size=svc_cache_size, coef0=svc_coef0,
                      degree=svc_degree, gamma=svc_gamma, kernel=svc_kernel)

        title_info = 'SVC soft fit:'
        print "\n" + title_info
        print 25 * "-"
        mysvc.fit(trX, trY)
        mysvc.load_data(trX, trY, tsX, tsY)
        print_output(mysvc, tsX, tsY, title_info)

    elif user_in == 3:
        print "(3) SVC: Testing 2-dimensional circular data \n"
        # Testing CIRCLE
        trX = np.array([[1, 1], [4, 1], [1, 4], [4, 4], [2, 2], [2, 3], [3, 2]])
        trY = [1, 1, 1, 1, -1, -1, -1]
        tsX = np.array([[0, 2], [3, 3], [6, 3]])
        tsY = [1, -1, 1]
        # kernel = 'poly'; degree = 2; gamma = 1; coef0 = 1; svc_C = 0.10000  # one point misclass

        svc_kernel = 'poly'
        svc_degree = 2
        svc_gamma = 1
        svc_coef0 = 1
        svc_cache_size = 200
        svc_C = 10000.0

        # use: print mysvc
        print "kernel = %s, degree = %d, gamma = %3.2f, coef0 = %3.2f, svc_C = %5.4f" \
              % (svc_kernel, svc_degree, svc_gamma, svc_coef0, svc_C)
        print "-----------------------------------------------------"

        mysvc = MySVC(Csoft=svc_C, cache_size=svc_cache_size, coef0=svc_coef0,
                      degree=svc_degree, gamma=svc_gamma, kernel=svc_kernel)

        title_info = 'SVC soft fit:'
        print "\n" + title_info
        print 25 * "-"
        mysvc.fit(trX, trY)
        mysvc.load_data(trX, trY, tsX, tsY)
        print_output(mysvc, tsX, tsY, title_info)

    elif user_in == 4:
        print "(4) SVC: Testing extended 2-dimensional circular data \n"
        # Testing extended CIRCLE
        trX = np.array([[1, 1], [4, 1], [1, 4], [4, 4], [2, 2], [2, 3], [3, 2], [5, 4.5]])
        trY = [1, 1, 1, 1, -1, -1, -1, -1]
        tsX = np.array([[0, 2], [3, 3], [6, 3]])
        tsY = [1, -1, 1]

        svc_kernel = 'rbf'
        svc_degree = 2
        svc_gamma = 0.1
        svc_coef0 = 1
        svc_cache_size = 200
        svc_C = 1000.

        # use: print mysvc
        print "kernel = %s, degree = %d, gamma = %3.2f, coef0 = %3.2f, svc_C = %5.4f" \
              % (svc_kernel, svc_degree, svc_gamma, svc_coef0, svc_C)
        print "-----------------------------------------------------"

        mysvc = MySVC(Csoft=svc_C, cache_size=svc_cache_size, coef0=svc_coef0,
                      degree=svc_degree, gamma=svc_gamma, kernel=svc_kernel)

        title_info = 'SVC soft fit:'
        print "\n" + title_info
        print 25 * "-"
        mysvc.fit(trX, trY)
        mysvc.load_data(trX, trY, tsX, tsY)
        print_output(mysvc, tsX, tsY, title_info)

    elif user_in == 5:
        print "(5) SVC: IRIS dataset: Testing 2-attribute, 2-class version (samples 0,...,99, classes 0, 1) \n"
        # Testing 2 attribute IRIS
        iris = datasets.load_iris()
        trX = iris.data[:100, :]
        # Classes 0 and 1 only
        trX = trX[:, [0, 1]]
        trY = iris.target[:100]
        trY = [i if i == 1 else -1 for i in trY]

        svc_kernel = 'rbf'
        svc_degree = 1
        svc_gamma = 1
        svc_coef0 = 1
        svc_cache_size = 200
        svc_C = 0.10000

        # use: print mysvc
        print "\nkernel = %s, degree = %d, gamma = %3.2f, coef0 = %3.2f, Csoft = %5.4f"\
              % (svc_kernel, svc_degree, svc_gamma, svc_coef0, svc_C)
        print "-----------------------------------------------------"
        mysvc = MySVC(Csoft=svc_C, cache_size=svc_cache_size, coef0=svc_coef0,
                      degree=svc_degree, gamma=svc_gamma, kernel=svc_kernel)
        #####################################################################
        # FKC
        # kernel = linear, degree = 1, gamma = 1.00, coef0 = 1.00, Csoft = 10000.0000
        # fkc.fit() results in incorrect classification, fkc.fit_grb() is okay
        #####################################################################

        title_info = 'SVC soft fit:'
        print "\n" + title_info
        print 25 * "-"
        mysvc.fit(trX, trY)
        mysvc.load_data(trX, trY, tsX, tsY)  # TODO: Split load_data()
        print "tr_accuracy = ", mysvc.score_train()
        mysvc.plot2d(this_title_info=title_info)

    elif user_in == 6:
        print "(6) SVC: IRIS dataset: Testing 4-attribute, 2-class version (samples 0,...,99, classes 0, 1) \n"
        # Testing 4 attribute IRIS
        iris = datasets.load_iris()
        scaler = MinMaxScaler()
        trX = scaler.fit_transform(iris.data[:100, :])
        trY = iris.target[:100]
        trY = [i if i == 1 else -1 for i in trY]

        svc_kernel = 'poly'
        svc_degree = 4
        svc_gamma = 1
        svc_coef0 = 1
        svc_cache_size = 200
        svc_C = 10000

        # use: print mysvc
        print "\nkernel = %s, degree = %d, gamma = %3.2f, coef0 = %3.2f, Csoft = %5.4f"\
              % (svc_kernel, svc_degree, svc_gamma, svc_coef0, svc_C)
        print "-----------------------------------------------------"
        mysvc = MySVC(Csoft=svc_C, cache_size=svc_cache_size, coef0=svc_coef0,
                      degree=svc_degree, gamma=svc_gamma, kernel=svc_kernel)

        title_info = 'SVC soft fit:'
        print "\n" + title_info
        print 25 * "-"
        mysvc.fit(trX, trY)
        mysvc.load_data(trX, trY, tsX, tsY)
        print "tr_accuracy = ", mysvc.score_train()
        mysvc.plot2d(this_title_info=title_info)

    elif user_in == 7:
        print "(7) SVC: IRIS dataset: Computing generalization error for 2-class (100 experiments)"
        #####################################################################
        # Testing IRIS CLASSES 1 AND 2                                      #
        # Classes:              2                                           #
        # Samples per class:    50(Setosa), 50(versicolor)                 #
        # Samples total:        100                                         #
        # Dimensionality:       4                                          #
        # Features:             real, positive                              #
        #####################################################################
        myseed = 2
        np.random.seed(myseed)
        pd.set_option('expand_frame_repr', False)
        pd.set_option('display.max_rows', 400)

        # All inputs and all labels
        scaler = MinMaxScaler()
        iris_data = datasets.load_iris()
        df = pd.DataFrame(scaler.fit_transform(iris_data.data[:100, :]))
        y = iris_data.target[:100]
        y = np.array([i if i == 1 else -1 for i in y])

        svc_kernel = 'poly'
        svc_degree = 2
        svc_gamma = 1
        svc_coef0 = 1
        svc_C = 10
        svc_cache_size = 200

        clf = svm.SVC(C=svc_C, cache_size=svc_cache_size, coef0=svc_coef0,
                      degree=svc_degree, gamma=svc_gamma, kernel=svc_kernel)

        # use: print mysvc
        t = time()
        svc_gen_error_list = []
        num_experiments = 100
        print "Running %d experiments... \n" % num_experiments
        for i in range(num_experiments):
            print "Running experiment: %d" % (i + 1)
            # 67 train and 33 test samples
            trX, tsX, trY, tsY = train_test_split(df, y, test_size=0.33)
            (num_test_samples, num_features) = tsX.as_matrix().shape

            clf.fit(trX.as_matrix(), trY)
            ftest = clf.predict(tsX.as_matrix())
            num_wrong = 1 * (ftest != tsY).sum()
            svc_gen_error_list.append(num_wrong / float(num_test_samples))

        svc_gen_error = np.array(svc_gen_error_list).mean()
        if svc_kernel == 'rbf':
            pathname = 'C:\\Users\\amalysch\\PycharmProjects\\ml_center_repository\\' \
                       'ml_center_project\\ml_center_results\\'
            filename = 'svc_iris12_%s_gamma_%d_coef_%d_svc_C_%4.4f_seed_%d_num_exper_%d_gen_error_%0.4f.pickle' \
                       % (svc_kernel, svc_gamma, svc_coef0, svc_C, myseed, num_experiments, svc_gen_error)
            f = open(pathname + filename, 'w')
            pickle.dump(svc_gen_error_list, f)
            f.close()
        elif svc_kernel == 'poly':
            pathname = 'C:\\Users\\amalysch\\PycharmProjects\\ml_center_repository\\' \
                       'ml_center_project\\ml_center_results\\'
            filename = 'svc_iris12_%s_degree_%d_coef_%d_svc_C_%4.4f_seed_%d_num_exper_%d_gen_error_%0.4f.pickle' \
                       % (svc_kernel, svc_degree, svc_coef0, svc_C, myseed, num_experiments, svc_gen_error)
            f = open(pathname + filename, 'w')
            pickle.dump(svc_gen_error_list, f)
            f.close()
        else:
            # Linear kernel: degree = 1, coef0 = 0
            pathname = 'C:\\Users\\amalysch\\PycharmProjects\\ml_center_repository\\' \
                       'ml_center_project\\ml_center_results\\'
            filename = 'svc_iris12_%s_degree_1_coef_0_svc_C_%4.4f_seed_%d_num_exper_%d_gen_error_%0.4f.pickle' \
                       % (svc_kernel, myseed, svc_C, num_experiments, svc_gen_error)
            f = open(pathname + filename, 'w')
            pickle.dump(svc_gen_error_list, f)
            f.close()

        print "Generalization error BREAST CANCER dataset (%d experiments): " % num_experiments, "\n", \
            np.array(svc_gen_error_list)
        print "\nAverage Generalization Error BREAST CANCER dataset (%d experiments): " \
              % num_experiments, "%.3f" % svc_gen_error
        print "Elapsed time %4.1f seconds." % (time() - t)
    elif user_in == 9:
        print "(9) SVC: BREAST CANCER dataset: Computing generalization error (100 experiments)"
        #####################################################################
        # Testing BREAST CANCER                                             #
        # Classes:              2                                           #
        # Samples per class:    212(Malignant), 357(Benign)                 #
        # Samples total:        569                                         #
        # Dimensionality:       30                                          #
        # Features:             real, positive                              #
        #####################################################################
        myseed = 2
        np.random.seed(myseed)
        pd.set_option('expand_frame_repr', False)
        pd.set_option('display.max_rows', 400)

        # All inputs and all labels
        scaler = MinMaxScaler()
        bc_data = datasets.load_breast_cancer()
        df = pd.DataFrame(scaler.fit_transform(bc_data.data))
        y = bc_data.target
        y = np.array([i if i == 1 else -1 for i in y])

        svc_kernel = 'rbf'
        svc_degree = 2
        svc_C = 10000
        # svc_kernel = 'poly'
        # svc_degree = 4
        # svc_C = 10
        svc_gamma = 1 / (2 * 3. * 3.)
        svc_coef0 = 1
        svc_cache_size = 200

        mysvc = MySVC(Csoft=svc_C, cache_size=svc_cache_size, coef0=svc_coef0,
                      degree=svc_degree, gamma=svc_gamma, kernel=svc_kernel)

        # use: print mysvc
        t = time()
        svc_gen_error_list = []
        num_experiments = 100
        print "Running %d experiments... \n" % num_experiments
        for i in range(num_experiments):
            print "Running experiment: %d" % (i+1)
            # 381 train and 188 test samples
            trX, tsX, trY, tsY = train_test_split(df, y, test_size=0.33)
            (num_test_samples, num_features) = tsX.as_matrix().shape

            mysvc.fit(trX.as_matrix(), trY)
            # mysvc.score() returns accuracy, want gen error
            svc_gen_error_list.append(1 - mysvc.score(tsX.as_matrix(), tsY))

        svc_gen_error = np.array(svc_gen_error_list).mean()
        if svc_kernel == 'rbf':
            pathname = 'C:\\Users\\amalysch\\PycharmProjects\\ml_center_repository\\' \
                        'ml_center_project\\ml_center_results\\'
            filename = 'svc_bc_%s_gamma_%d_coef_%d_svc_C_%4.4f_seed_%d_num_exper_%d_gen_error_%0.4f.pickle' \
                       % (svc_kernel, svc_gamma, svc_coef0, svc_C, myseed, num_experiments, svc_gen_error)
            f = open(pathname + filename, 'w')
            pickle.dump(svc_gen_error_list, f)
            f.close()
        elif svc_kernel == 'poly':
            pathname = 'C:\\Users\\amalysch\\PycharmProjects\\ml_center_repository\\' \
                        'ml_center_project\\ml_center_results\\'
            filename = 'svc_bc_%s_degree_%d_coef_%d_svc_C_%4.4f_seed_%d_num_exper_%d_gen_error_%0.4f.pickle' \
                       % (svc_kernel, svc_degree, svc_coef0, svc_C, myseed, num_experiments, svc_gen_error)
            f = open(pathname + filename, 'w')
            pickle.dump(svc_gen_error_list, f)
            f.close()
        else:
            # Linear kernel: degree = 1, coef0 = 0
            pathname = 'C:\\Users\\amalysch\\PycharmProjects\\ml_center_repository\\' \
                        'ml_center_project\\ml_center_results\\'
            filename = 'svc_bc_%s_degree_1_coef_0_svc_C_%4.4f_seed_%d_num_exper_%d_gen_error_%0.4f.pickle' \
                       % (svc_kernel, myseed, svc_C, num_experiments, svc_gen_error)
            f = open(pathname + filename, 'w')
            pickle.dump(svc_gen_error_list, f)
            f.close()

        print "Generalization error BREAST CANCER dataset (%d experiments): " % num_experiments, "\n", \
            np.array(svc_gen_error_list)
        print "\nAverage Generalization Error BREAST CANCER dataset (%d experiments): " \
              % num_experiments, "%.5f" % svc_gen_error
        print "\nAverage Accuracy BREAST CANCER dataset (%d experiments): " \
              % num_experiments, "%.5f" % (1 - svc_gen_error)
        print "Elapsed time %4.1f seconds." % (time() - t)
    elif user_in == 11:
        print "(11) SVC RBF: BREAST CANCER dataset: Computing accuracy for many (gamma, C) (takes a while)"
        #####################################################################
        # Testing BREAST CANCER                                             #
        # Classes:              2                                           #
        # Samples per class:    212(Malignant), 357(Benign)                 #
        # Samples total:        569                                         #
        # Dimensionality:       30                                          #
        # Features:             real, positive                              #
        #####################################################################
        myseed = 2
        np.random.seed(myseed)
        pd.set_option('expand_frame_repr', False)
        pd.set_option('display.max_rows', 400)

        # All inputs and all labels
        scaler = MinMaxScaler()
        bc_data = datasets.load_breast_cancer()
        df = pd.DataFrame(scaler.fit_transform(bc_data.data))
        y = bc_data.target
        y = np.array([i if i == 1 else -1 for i in y])

        svc_kernel = 'rbf'
        svc_degree = 2  # irrelevant
        svc_coef0 = 1  # irrelevant
        svc_cache_size = 200

        svc_C_list = [1e4, 1e2, 1e0, 1e-2, 1e-4]
        svc_gamma_list = [1/(2 * 3.**2), 1/(2 * 2.**2), 1/(2 * 1.**2),
                          1/(2 * 0.8**2), 1/(2 * 0.6**2), 1/(2 * 0.4**2)]
        accuracy_list = []

        for svc_gamma in svc_gamma_list:
            for svc_C in svc_C_list:
                mysvc = MySVC(Csoft=svc_C, cache_size=svc_cache_size, coef0=svc_coef0,
                              degree=svc_degree, gamma=svc_gamma, kernel=svc_kernel)
                # use: print mysvc
                t = time()
                svc_gen_error_list = []
                num_experiments = 100
                print "Running %d experiments... \n" % num_experiments
                for i in range(num_experiments):
                    print "Running experiment: %d" % (i+1)
                    # 381 train and 188 test samples
                    trX, tsX, trY, tsY = train_test_split(df, y, test_size=0.33)
                    (num_test_samples, num_features) = tsX.as_matrix().shape

                    mysvc.fit(trX.as_matrix(), trY)
                    # mysvc.score() returns accuracy, want gen error
                    svc_gen_error_list.append(1 - mysvc.score(tsX.as_matrix(), tsY))

                svc_gen_error = np.array(svc_gen_error_list).mean()
                if svc_kernel == 'rbf':
                    pathname = 'C:\\Users\\amalysch\\PycharmProjects\\ml_center_repository\\' \
                                'ml_center_project\\ml_center_results\\'
                    filename = 'svc_bc_%s_gamma_%d_coef_%d_svc_C_%4.4f_seed_%d_num_exper_%d_gen_error_%0.4f.pickle' \
                               % (svc_kernel, svc_gamma, svc_coef0, svc_C, myseed, num_experiments, svc_gen_error)
                    f = open(pathname + filename, 'w')
                    pickle.dump(svc_gen_error_list, f)
                    f.close()
                elif svc_kernel == 'poly':
                    pathname = 'C:\\Users\\amalysch\\PycharmProjects\\ml_center_repository\\' \
                                'ml_center_project\\ml_center_results\\'
                    filename = 'svc_bc_%s_degree_%d_coef_%d_svc_C_%4.4f_seed_%d_num_exper_%d_gen_error_%0.4f.pickle' \
                               % (svc_kernel, svc_degree, svc_coef0, svc_C, myseed, num_experiments, svc_gen_error)
                    f = open(pathname + filename, 'w')
                    pickle.dump(svc_gen_error_list, f)
                    f.close()
                else:
                    # Linear kernel: degree = 1, coef0 = 0
                    pathname = 'C:\\Users\\amalysch\\PycharmProjects\\ml_center_repository\\' \
                                'ml_center_project\\ml_center_results\\'
                    filename = 'svc_bc_%s_degree_1_coef_0_svc_C_%4.4f_seed_%d_num_exper_%d_gen_error_%0.4f.pickle' \
                               % (svc_kernel, myseed, svc_C, num_experiments, svc_gen_error)
                    f = open(pathname + filename, 'w')
                    pickle.dump(svc_gen_error_list, f)
                    f.close()

                print "Generalization error BREAST CANCER dataset (%d experiments): " % num_experiments, "\n", \
                    np.array(svc_gen_error_list)
                print "\nAverage Generalization Error BREAST CANCER dataset (%d experiments): " \
                      % num_experiments, "%.5f" % svc_gen_error
                print "\nAverage Accuracy BREAST CANCER dataset (%d experiments): " \
                      % num_experiments, "%.5f" % (1 - svc_gen_error)
                accuracy_list.append((1 - svc_gen_error))
                print "Elapsed time %4.1f seconds." % (time() - t)
        accuracy_array = np.array(accuracy_list)
        accuracy_array = accuracy_array.reshape(6, 5)
        print "accuracy_array = \n", accuracy_array
    elif user_in == 12:
        print "(12) SVC POLY: BREAST CANCER dataset: Computing accuracy for many (degree, C) (takes a while)"
        #####################################################################
        # Testing BREAST CANCER                                             #
        # Classes:              2                                           #
        # Samples per class:    212(Malignant), 357(Benign)                 #
        # Samples total:        569                                         #
        # Dimensionality:       30                                          #
        # Features:             real, positive                              #
        #####################################################################
        myseed = 2
        np.random.seed(myseed)
        pd.set_option('expand_frame_repr', False)
        pd.set_option('display.max_rows', 400)

        # All inputs and all labels
        scaler = MinMaxScaler()
        bc_data = datasets.load_breast_cancer()
        df = pd.DataFrame(scaler.fit_transform(bc_data.data))
        y = bc_data.target
        y = np.array([i if i == 1 else -1 for i in y])

        svc_kernel = 'poly'
        svc_gamma = 1
        svc_coef0 = 1
        svc_cache_size = 200

        svc_C_list = [1e4, 1e2, 1e0, 1e-2, 1e-4]
        svc_degree_list = [1, 2, 3, 4, 5]
        accuracy_list = []

        for svc_degree in svc_degree_list:
            for svc_C in svc_C_list:
                mysvc = MySVC(Csoft=svc_C, cache_size=svc_cache_size, coef0=svc_coef0,
                              degree=svc_degree, gamma=svc_gamma, kernel=svc_kernel)
                # use: print mysvc
                t = time()
                svc_gen_error_list = []
                num_experiments = 100
                print "Running %d experiments... \n" % num_experiments
                for i in range(num_experiments):
                    print "Running experiment: %d" % (i + 1)
                    # 381 train and 188 test samples
                    trX, tsX, trY, tsY = train_test_split(df, y, test_size=0.33)
                    (num_test_samples, num_features) = tsX.as_matrix().shape

                    mysvc.fit(trX.as_matrix(), trY)
                    # mysvc.score() returns accuracy, want gen error
                    svc_gen_error_list.append(1 - mysvc.score(tsX.as_matrix(), tsY))

                svc_gen_error = np.array(svc_gen_error_list).mean()
                if svc_kernel == 'rbf':
                    pathname = 'C:\\Users\\amalysch\\PycharmProjects\\ml_center_repository\\' \
                               'ml_center_project\\ml_center_results\\'
                    filename = 'svc_bc_%s_gamma_%d_coef_%d_svc_C_%4.4f_seed_%d_num_exper_%d_gen_error_%0.4f.pickle' \
                               % (svc_kernel, svc_gamma, svc_coef0, svc_C, myseed, num_experiments, svc_gen_error)
                    f = open(pathname + filename, 'w')
                    pickle.dump(svc_gen_error_list, f)
                    f.close()
                elif svc_kernel == 'poly':
                    pathname = 'C:\\Users\\amalysch\\PycharmProjects\\ml_center_repository\\' \
                               'ml_center_project\\ml_center_results\\'
                    filename = 'svc_bc_%s_degree_%d_coef_%d_svc_C_%4.4f_seed_%d_num_exper_%d_gen_error_%0.4f.pickle' \
                               % (svc_kernel, svc_degree, svc_coef0, svc_C, myseed, num_experiments, svc_gen_error)
                    f = open(pathname + filename, 'w')
                    pickle.dump(svc_gen_error_list, f)
                    f.close()
                else:
                    # Linear kernel: degree = 1, coef0 = 0
                    pathname = 'C:\\Users\\amalysch\\PycharmProjects\\ml_center_repository\\' \
                               'ml_center_project\\ml_center_results\\'
                    filename = 'svc_bc_%s_degree_1_coef_0_svc_C_%4.4f_seed_%d_num_exper_%d_gen_error_%0.4f.pickle' \
                               % (svc_kernel, myseed, svc_C, num_experiments, svc_gen_error)
                    f = open(pathname + filename, 'w')
                    pickle.dump(svc_gen_error_list, f)
                    f.close()

                print "Generalization error BREAST CANCER dataset (%d experiments): " % num_experiments, "\n", \
                    np.array(svc_gen_error_list)
                print "\nAverage Generalization Error BREAST CANCER dataset (%d experiments): " \
                      % num_experiments, "%.5f" % svc_gen_error
                print "\nAverage Accuracy BREAST CANCER dataset (%d experiments): " \
                      % num_experiments, "%.5f" % (1 - svc_gen_error)
                accuracy_list.append((1 - svc_gen_error))
                print "Elapsed time %4.1f seconds." % (time() - t)
        accuracy_array = np.array(accuracy_list)
        accuracy_array = accuracy_array.reshape(5, 5)
        print "accuracy_array = \n", accuracy_array
    else:
        print "Invalid selection. Program terminating. "
